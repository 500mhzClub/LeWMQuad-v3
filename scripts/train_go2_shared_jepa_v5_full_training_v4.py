#!/usr/bin/env python3
"""New-process exact trainer/publisher for Shared JEPA V5 full training V4.

The standard-library bootstrap validates the retained exact reservation and
opens the bound preflight receipt before any other repository input.  Torch,
the V4/V5 model sources, checkpoints, RGB, and role payload are imported or
opened only by the fixed production backend after that boundary.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lewm.benchmarks import go2_shared_jepa_v5_full_training_v4_policy as policy


if __name__ == "__main__":
    ROOT = SCRIPT_ROOT

    def _directory_flags() -> int:
        flags = getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
            raise PermissionError("exact trainer requires no-follow directories")
        return os.O_RDONLY | flags | getattr(os, "O_CLOEXEC", 0)


    def _file_flags() -> int:
        if not getattr(os, "O_NOFOLLOW", 0):
            raise PermissionError("exact trainer requires no-follow files")
        return (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )


    def _file_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_mode),
            int(metadata.st_nlink),
            int(metadata.st_uid),
            int(metadata.st_gid),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )


    def _read_fd_all(descriptor: int, *, name: str) -> bytes:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PermissionError(f"{name} is not a singly linked regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _file_fingerprint(before) != _file_fingerprint(after):
            raise RuntimeError(f"{name} changed while read")
        return b"".join(chunks)


    def _read_repo_relative(relative: str) -> bytes:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise PermissionError("exact trainer input path escaped")
        root_fd = os.open(ROOT, _directory_flags())
        descriptors = [root_fd]
        parent_fd = root_fd
        try:
            for component in path.parts[:-1]:
                descriptor = os.open(component, _directory_flags(), dir_fd=parent_fd)
                descriptors.append(descriptor)
                parent_fd = descriptor
            descriptor = os.open(path.name, _file_flags(), dir_fd=parent_fd)
            try:
                return _read_fd_all(descriptor, name=relative)
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


    def _read_claim_relative(claim_fd: int, relative: str) -> bytes:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise PermissionError("exact claim path escaped")
        descriptors: list[int] = []
        parent_fd = claim_fd
        try:
            for component in path.parts[:-1]:
                descriptor = os.open(component, _directory_flags(), dir_fd=parent_fd)
                descriptors.append(descriptor)
                parent_fd = descriptor
            descriptor = os.open(path.name, _file_flags(), dir_fd=parent_fd)
            try:
                return _read_fd_all(descriptor, name=relative)
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


    @dataclass
    class TrainerContext:
        claim_fd: int
        parent_fd: int
        directory_name: str
        directory_identity: tuple[int, int]
        reservation: Mapping[str, Any]
        reservation_raw: bytes
        manifest: Mapping[str, Any]
        manifest_raw: bytes
        preflight_receipt: Mapping[str, Any]
        preflight_receipt_raw: bytes
        raw_manifest: Mapping[str, Any]
        raw_manifest_raw: bytes
        raw_audit: Mapping[str, Any]
        raw_audit_raw: bytes
        v4_ladder: Mapping[str, Any]
        v4_ladder_raw: bytes
        source_review: Mapping[str, Any]
        input_bindings: Mapping[str, Any]
        events: list[dict[str, Any]] = field(default_factory=list)
        published: dict[str, dict[str, Any]] = field(default_factory=dict)


    def _assert_claim(
        claim_fd: int,
        parent_fd: int,
        directory_name: str,
    ) -> tuple[int, int]:
        opened = os.fstat(claim_fd)
        named = os.stat(directory_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise PermissionError("exact trainer claim identity changed")
        return int(opened.st_dev), int(opened.st_ino)


    def _load_reservation(
        claim_fd: int,
        parent_fd: int,
        directory_name: str,
        manifest_file_sha256: str,
        manifest_content_sha256: str,
    ) -> tuple[dict[str, Any], bytes, tuple[int, int]]:
        identity = _assert_claim(claim_fd, parent_fd, directory_name)
        raw = _read_claim_relative(claim_fd, "reservation.json")
        value = policy.parse_canonical_json(raw, name="exact reservation")
        if (
            value.get("schema") != policy.EXACT_RESERVATION_SCHEMA
            or value.get("status")
            != "reserved_before_torch_model_checkpoint_or_payload"
            or value.get("directory_identity") != list(identity)
            or value.get("final_exact_authorization_file_sha256") != manifest_file_sha256
            or value.get("final_exact_authorization_content_sha256")
            != manifest_content_sha256
            or value.get("torch_or_payload_opened_before_reservation") is not False
            or value.get("retry_authorized") is not False
        ):
            raise PermissionError("exact trainer reservation contract changed")
        bindings = value.get("required_exact_bindings")
        if not isinstance(bindings, Mapping) or set(bindings) != set(
            policy.FINAL_REQUIRED_BINDING_NAMES
        ) or any(not policy.is_sha256(item) for item in bindings.values()):
            raise PermissionError("exact trainer reservation bindings are incomplete")
        return value, raw, identity


    def _append_event(
        events: list[dict[str, Any]],
        *,
        stage: str,
        arm: str | None,
        role: str,
        operation: str,
        relative_path: str,
        expected_sha256: str,
        raw: bytes,
    ) -> None:
        observed = hashlib.sha256(raw).hexdigest()
        events.append(
            policy.append_access_event(
                events,
                stage=stage,
                arm=arm,
                role=role,
                operation=operation,
                relative_path=relative_path,
                expected_sha256=expected_sha256,
                observed_sha256=observed,
                byte_count=len(raw),
                process_identity=str(os.getpid()),
            )
        )


    def _read_bound_repo_input(
        events: list[dict[str, Any]],
        *,
        relative_path: str,
        expected_sha256: str,
        stage: str,
        role: str,
        arm: str | None = None,
        operation: str = "read_and_rehash",
    ) -> bytes:
        if not policy.is_sha256(expected_sha256):
            raise ValueError("exact bound input SHA-256 is malformed")
        raw = _read_repo_relative(relative_path)
        _append_event(
            events,
            stage=stage,
            arm=arm,
            role=role,
            operation=operation,
            relative_path=relative_path,
            expected_sha256=expected_sha256,
            raw=raw,
        )
        return raw


    def _parse_content_json(raw: bytes, *, name: str) -> dict[str, Any]:
        return policy.parse_canonical_json(raw, name=name)


    def _load_preflight_first(
        reservation: Mapping[str, Any],
        events: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], bytes]:
        bindings = reservation["required_exact_bindings"]
        raw = _read_bound_repo_input(
            events,
            relative_path=policy.PREFLIGHT_RECEIPT_RELATIVE_PATH,
            expected_sha256=bindings["preflight_receipt_file_sha256"],
            stage="exact_input",
            role="preflight_receipt",
            operation="first_post_reservation_input_open",
        )
        receipt = _parse_content_json(raw, name="bound preflight receipt")
        if (
            receipt.get("schema") != policy.PREFLIGHT_RECEIPT_SCHEMA
            or receipt.get("status") != "PASS"
            or receipt.get("payload_open_count") != 0
            or receipt.get("forbidden_open_count") != 0
            or receipt.get("device_contract") != policy.DEVICE_CONTRACT
        ):
            raise PermissionError("bound preflight receipt did not pass")
        return receipt, raw


    def _validate_raw_v13_authority_chain(
        source_raw_by_path: Mapping[str, bytes],
    ) -> dict[str, Any]:
        """Use the policy's independently reconstructed exact Raw V13 schemas."""

        paths = (
            policy.RAW_BUILDER_V9_REVIEW_RELATIVE_PATH,
            policy.RAW_AUDITOR_V13_REVIEW_RELATIVE_PATH,
            policy.RAW_AUDITOR_V13_AUTHORIZATION_RELATIVE_PATH,
            policy.RAW_AUDITOR_V13_FINGERPRINT_RELATIVE_PATH,
        )
        if any(path not in source_raw_by_path for path in paths):
            raise PermissionError("terminal Raw V13 authority chain is incomplete")
        return policy.validate_raw_v13_source_chain(
            builder_review=_parse_content_json(
                source_raw_by_path[paths[0]],
                name="Builder V9 independent review",
            ),
            auditor_review=_parse_content_json(
                source_raw_by_path[paths[1]],
                name="Auditor V13 independent review",
            ),
            authorization=_parse_content_json(
                source_raw_by_path[paths[2]],
                name="Auditor V13 authorization",
            ),
            fingerprint=_parse_content_json(
                source_raw_by_path[paths[3]],
                name="Auditor V13 authorization fingerprint",
            ),
        )


    def _load_authority_after_preflight(
        reservation: Mapping[str, Any],
        manifest_file_sha256: str,
        manifest_content_sha256: str,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        bindings = reservation["required_exact_bindings"]
        manifest_raw = _read_bound_repo_input(
            events,
            relative_path=policy.FINAL_EXACT_EXECUTION_AUTHORIZATION_RELATIVE_PATH,
            expected_sha256=manifest_file_sha256,
            stage="exact_source_closure",
            role="final_exact_authorization",
        )
        manifest = policy.validate_final_exact_execution_authorization(
            _parse_content_json(manifest_raw, name="final exact authorization in trainer"),
        )
        if (
            manifest["content_sha256"] != manifest_content_sha256
            or manifest["required_exact_bindings"] != bindings
        ):
            raise PermissionError("reserved manifest and trainer manifest differ")

        implementation_review_raw = _read_bound_repo_input(
            events,
            relative_path=policy.IMPLEMENTATION_REVIEW_RELATIVE_PATH,
            expected_sha256=bindings["implementation_independent_review_file_sha256"],
            stage="exact_source_closure",
            role="implementation_review",
        )
        implementation_review = policy.validate_implementation_review(
            _parse_content_json(
                implementation_review_raw,
                name="implementation independent review",
            )
        )
        implementation_binding_names = {
            policy.POLICY_RELATIVE_PATH: "implementation_policy_source_sha256",
            policy.LOSS_ADAPTER_RELATIVE_PATH: "loss_adapter_source_sha256",
            policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH: "preflight_executor_source_sha256",
            policy.PREFLIGHT_VERIFIER_RELATIVE_PATH: "preflight_verifier_source_sha256",
            policy.EXACT_EXECUTOR_RELATIVE_PATH: "exact_executor_source_sha256",
            policy.EXACT_TRAINER_RELATIVE_PATH: "exact_trainer_source_sha256",
            policy.EXACT_VERIFIER_RELATIVE_PATH: "exact_verifier_source_sha256",
        }
        if any(
            implementation_review["reviewed_production_sources"].get(relative)
            != bindings[binding_name]
            for relative, binding_name in implementation_binding_names.items()
        ):
            raise PermissionError("implementation review and exact bindings differ")
        source_hashes: dict[str, str] = {}
        source_raw_by_path: dict[str, bytes] = {}
        for relative, expected in {
            **policy.reviewed_source_bindings(),
            **dict(implementation_review["reviewed_production_sources"]),
            policy.RAW_SUPERVISION_BUILDER_RELATIVE_PATH: bindings[
                "development_raw_supervision_builder_source_sha256"
            ],
            policy.RAW_SUPERVISION_AUDITOR_RELATIVE_PATH: bindings[
                "development_raw_supervision_auditor_source_sha256"
            ],
        }.items():
            raw = _read_bound_repo_input(
                events,
                relative_path=relative,
                expected_sha256=expected,
                stage="exact_source_closure",
                role="source_closure",
            )
            source_hashes[relative] = hashlib.sha256(raw).hexdigest()
            source_raw_by_path[relative] = raw
        raw_v13_chain = _validate_raw_v13_authority_chain(source_raw_by_path)

        completed_raw = _read_bound_repo_input(
            events,
            relative_path=policy.PREFLIGHT_COMPLETED_RELATIVE_PATH,
            expected_sha256=bindings["preflight_completed_file_sha256"],
            stage="exact_input",
            role="preflight_receipt",
        )
        completed = _parse_content_json(completed_raw, name="preflight completion")
        receipt_binding = completed.get("artifacts_before_completion", {}).get(
            "gpu_smoke_receipt.json"
        )
        if (
            completed.get("schema") != policy.PREFLIGHT_COMPLETION_SCHEMA
            or completed.get("status")
            != "completed_after_independent_reconstruction"
            or not isinstance(receipt_binding, Mapping)
            or receipt_binding.get("file_sha256")
            != bindings["preflight_receipt_file_sha256"]
        ):
            raise PermissionError("preflight completion does not bind the receipt")
        preflight_review_raw = _read_bound_repo_input(
            events,
            relative_path=policy.PREFLIGHT_INDEPENDENT_REVIEW_RELATIVE_PATH,
            expected_sha256=bindings["preflight_independent_review_file_sha256"],
            stage="exact_source_closure",
            role="source_closure",
        )

        raw_manifest_raw = _read_bound_repo_input(
            events,
            relative_path=policy.RAW_SUPERVISION_MANIFEST_RELATIVE_PATH,
            expected_sha256=bindings[
                "development_raw_supervision_manifest_file_sha256"
            ],
            stage="exact_input",
            role="raw_supervision_manifest",
        )
        raw_manifest = policy.validate_raw_v13_manifest(
            _parse_content_json(raw_manifest_raw, name="raw manifest")
        )
        raw_audit_raw = _read_bound_repo_input(
            events,
            relative_path=policy.RAW_SUPERVISION_AUDIT_RELATIVE_PATH,
            expected_sha256=bindings["development_raw_supervision_audit_file_sha256"],
            stage="exact_input",
            role="raw_supervision_audit",
        )
        raw_audit = policy.validate_raw_v13_terminal_report(
            _parse_content_json(raw_audit_raw, name="raw audit")
        )
        if (
            manifest["raw_v13_dataset_use_grant"] != policy.RAW_DATASET_USE_GRANT
            or manifest["authority"] != policy.FINAL_EXACT_AUTHORITY
        ):
            raise PermissionError("final authorization did not activate the narrow Raw V13 grant")

        camera_sources = manifest["camera_v14_source_bindings"]
        ladder_rows = policy.validate_camera_v14_ladder_rows(
            manifest["camera_ladder_rows"],
            reviewed_source_bindings=camera_sources,
        )
        if (
            bindings["camera_v14_source_review_file_sha256"]
            != ladder_rows[0]["source_review"]["file_sha256"]
            or bindings["camera_v14_source_review_content_sha256"]
            != ladder_rows[0]["source_review"]["content_sha256"]
            or bindings["camera_v14_n5_gate_pass_file_sha256"]
            != ladder_rows[0]["gate"]["file_sha256"]
            or bindings["camera_v14_n5_gate_pass_content_sha256"]
            != ladder_rows[0]["gate"]["content_sha256"]
            or bindings["v4_primary_seed_20260710_n320_checkpoint_file_sha256"]
            != ladder_rows[3]["checkpoint"]["file_sha256"]
        ):
            raise PermissionError("Camera V14 row bindings disagree with final authorization")

        for relative, expected in camera_sources.items():
            _read_bound_repo_input(
                events,
                relative_path=relative,
                expected_sha256=expected,
                stage="exact_source_closure",
                role="source_closure",
            )
        camera_source_review_raw = _read_bound_repo_input(
            events,
            relative_path=policy.CAMERA_V14_SOURCE_REVIEW_RELATIVE_PATH,
            expected_sha256=bindings["camera_v14_source_review_file_sha256"],
            stage="exact_source_closure",
            role="source_closure",
        )
        camera_source_review = _parse_content_json(
            camera_source_review_raw,
            name="Camera V14 source review",
        )
        if (
            camera_source_review.get("content_sha256")
            != bindings["camera_v14_source_review_content_sha256"]
            or camera_source_review.get("source_closure_approved") is not True
            or camera_source_review.get("scientific_retry_authorized") is not False
        ):
            raise PermissionError("Camera V14 source review did not pass")

        ladder_preregistration_raw = _read_bound_repo_input(
            events,
            relative_path=policy.CAMERA_V14_LADDER_PREREGISTRATION_RELATIVE_PATH,
            expected_sha256=bindings["camera_v14_ladder_preregistration_file_sha256"],
            stage="exact_source_closure",
            role="source_closure",
        )
        ladder_review_raw = _read_bound_repo_input(
            events,
            relative_path=policy.CAMERA_V14_LADDER_REVIEW_RELATIVE_PATH,
            expected_sha256=bindings["camera_v14_ladder_independent_review_file_sha256"],
            stage="exact_source_closure",
            role="source_closure",
        )
        ladder_review = _parse_content_json(
            ladder_review_raw,
            name="Camera V14 ladder independent review",
        )
        if ladder_review.get("verdict") != "PASS":
            raise PermissionError("Camera V14 ladder review did not pass")

        rung_values: list[dict[str, Any]] = []
        for row in ladder_rows:
            parsed: dict[str, Any] = {}
            for name in ("reservation", "source_review", "gate", "completion", "rung_review"):
                binding = row[name]
                raw = _read_bound_repo_input(
                    events,
                    relative_path=binding["path"],
                    expected_sha256=binding["file_sha256"],
                    stage="exact_input",
                    role="camera_v14_two_seed_ladder",
                )
                value = _parse_content_json(
                    raw,
                    name=f"Camera V14 row {row['row_index']} {name}",
                )
                if value.get("content_sha256") != binding["content_sha256"]:
                    raise PermissionError("Camera V14 rung content binding changed")
                parsed[name] = value
            checkpoint_binding = row["checkpoint"]
            _read_bound_repo_input(
                events,
                relative_path=checkpoint_binding["path"],
                expected_sha256=checkpoint_binding["file_sha256"],
                stage="exact_input",
                role=(
                    "camera_v14_primary_checkpoint"
                    if row["row_index"] == 3
                    else "camera_v14_two_seed_ladder"
                ),
            )
            if (
                parsed["gate"].get("passes") is not True
                or parsed["rung_review"].get("verdict") != "PASS"
                or parsed["completion"].get("content_sha256")
                != row["completion"]["content_sha256"]
            ):
                raise PermissionError("Camera V14 rung evidence did not independently pass")
            rung_values.append(parsed)

        camera_n5_gate = rung_values[0]["gate"]
        camera_n5_gate_raw = policy.canonical_json_bytes(camera_n5_gate) + b"\n"
        v4_ladder_raw = _read_bound_repo_input(
            events,
            relative_path=policy.CAMERA_V14_TWO_SEED_LADDER_RELATIVE_PATH,
            expected_sha256=bindings["camera_v14_two_seed_ladder_pass_file_sha256"],
            stage="exact_input",
            role="camera_v14_two_seed_ladder",
        )
        v4_ladder = _parse_content_json(v4_ladder_raw, name="Camera V14 aggregate gate")
        aggregate = manifest["camera_ladder_aggregate"]
        if (
            v4_ladder.get("content_sha256")
            != bindings["camera_v14_two_seed_ladder_pass_content_sha256"]
            or aggregate.get("rows_sha256") != policy.canonical_json_sha256(ladder_rows)
            or aggregate.get("ordered_rung_count") != 8
            or aggregate.get("additional_attempt_count") != 7
            or aggregate.get("seed_20260710_n5_reexecuted") is not False
            or aggregate.get("retry_performed") is not False
        ):
            raise PermissionError("Camera V14 aggregate ladder evidence changed")
        return {
            "manifest": manifest,
            "manifest_raw": manifest_raw,
            "implementation_review": implementation_review,
            "implementation_review_raw": implementation_review_raw,
            "source_hashes": source_hashes,
            "raw_v13_chain": raw_v13_chain,
            "preflight_completed": completed,
            "preflight_completed_raw": completed_raw,
            "preflight_review_raw": preflight_review_raw,
            "raw_manifest": raw_manifest,
            "raw_manifest_raw": raw_manifest_raw,
            "raw_audit": raw_audit,
            "raw_audit_raw": raw_audit_raw,
            "camera_source_review": camera_source_review,
            "camera_source_review_raw": camera_source_review_raw,
            "camera_n5_gate": camera_n5_gate,
            "camera_n5_gate_raw": camera_n5_gate_raw,
            "ladder_preregistration_raw": ladder_preregistration_raw,
            "ladder_review": ladder_review,
            "ladder_review_raw": ladder_review_raw,
            "v4_ladder": v4_ladder,
            "v4_ladder_raw": v4_ladder_raw,
        }


    def _mkdirs_at(claim_fd: int, relative_parent: Path) -> int:
        parent_fd = os.dup(claim_fd)
        try:
            for component in relative_parent.parts:
                if component in {"", ".", ".."}:
                    raise PermissionError("exact publication directory escaped")
                try:
                    os.mkdir(component, 0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except FileExistsError:
                    pass
                child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = child_fd
            return parent_fd
        except BaseException:
            os.close(parent_fd)
            raise


    def _write_relative_exclusive(claim_fd: int, relative: str, raw: bytes) -> None:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise PermissionError("exact publication path escaped")
        parent_fd = _mkdirs_at(claim_fd, path.parent)
        try:
            descriptor = os.open(
                path.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_fd,
            )
            try:
                view = memoryview(raw)
                while view:
                    count = os.write(descriptor, view)
                    if count <= 0:
                        raise OSError("exact publication write made no progress")
                    view = view[count:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


    def _publish_json(
        context: TrainerContext,
        relative: str,
        core: Mapping[str, Any],
    ) -> dict[str, Any]:
        copied = dict(core)
        if "content_sha256" in copied:
            declared = copied["content_sha256"]
            unhashed = dict(copied)
            unhashed.pop("content_sha256")
            if (
                not policy.is_sha256(declared)
                or policy.canonical_json_sha256(unhashed) != declared
            ):
                raise ValueError("prehashed exact JSON artifact changed")
            value = copied
        else:
            value = policy.content_value(copied)
        raw = policy.canonical_json_bytes(value) + b"\n"
        _write_relative_exclusive(context.claim_fd, relative, raw)
        binding = policy.artifact_binding(
            relative,
            raw,
            content_sha256=str(value["content_sha256"]),
        )
        context.published[relative] = binding
        return value


    def _publish_bytes(
        context: TrainerContext,
        relative: str,
        raw: bytes,
        *,
        content_sha256: str | None = None,
        state_sha256: str | None = None,
    ) -> dict[str, Any]:
        _write_relative_exclusive(context.claim_fd, relative, raw)
        binding = policy.artifact_binding(
            relative,
            raw,
            content_sha256=content_sha256,
        )
        if state_sha256 is not None:
            if not policy.is_sha256(state_sha256):
                raise ValueError("exact state SHA-256 is malformed")
            binding["state_sha256"] = state_sha256
        context.published[relative] = binding
        return binding


    def _publish_bootstrap_artifacts(context: TrainerContext) -> None:
        _publish_json(
            context,
            "source_review.json",
            dict(context.source_review),
        )
        _publish_json(
            context,
            "input_bindings.json",
            dict(context.input_bindings),
        )
        _publish_json(
            context,
            "preflight_receipt_binding.json",
            {
                "schema": "lewm_go2_shared_jepa_v5_full_training_v4_preflight_binding_v1",
                "receipt": policy.artifact_binding(
                    policy.PREFLIGHT_RECEIPT_RELATIVE_PATH,
                    context.preflight_receipt_raw,
                    content_sha256=str(context.preflight_receipt["content_sha256"]),
                ),
                "first_post_reservation_input_open": True,
                "preflight_live_state_inherited": False,
                "preflight_rerun": False,
            },
        )


    def _parse_jsonl(raw: bytes, *, name: str) -> list[dict[str, Any]]:
        if not raw or not raw.endswith(b"\n") or b"\n\n" in raw:
            raise ValueError(f"{name} is not canonical nonempty JSONL")
        result: list[dict[str, Any]] = []
        for index, line in enumerate(raw.splitlines(), start=1):
            value = json.loads(line.decode("ascii"))
            if not isinstance(value, dict) or policy.canonical_json_bytes(value) != line:
                raise ValueError(f"{name} row {index} is noncanonical")
            core = dict(value)
            declared = core.pop("content_sha256", None)
            if not policy.is_sha256(declared) or policy.canonical_json_sha256(core) != declared:
                raise ValueError(f"{name} row {index} content hash changed")
            result.append(value)
        return result


    def _fixed_production_backend_loader() -> Any:
        if (
            os.environ.get("HIP_VISIBLE_DEVICES") != "0"
            or os.environ.get("ROCR_VISIBLE_DEVICES") != "0"
            or "HSA_OVERRIDE_GFX_VERSION" in os.environ
        ):
            raise PermissionError("exact trainer accelerator environment changed")
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn.functional as F
        from lewm.benchmarks.go2_observable_camera_ray_fit_v4_metrics import (
            ObservableCameraRayFitV4MetricAccumulator,
        )
        from lewm.models.observable_camera_ray_evidence_v4 import (
            ObservableCameraRayEvidenceV4Model,
        )
        from lewm.models.observable_camera_ray_evidence_v4_training import (
            derive_observable_camera_ray_evidence_v4_targets,
            soft_rasterize_observable_camera_ray_evidence_v4,
        )
        from lewm.models import shared_observable_camera_ray_jepa_v5 as model_module
        from lewm.models import (
            shared_observable_camera_ray_jepa_v5_full_training_v4_loss
            as loss_adapter,
        )

        class FixedProductionTrainingBackend:
            def __init__(self, context: TrainerContext) -> None:
                self.context = context
                self.torch = torch
                self.np = np
                self.Image = Image
                self.F = F
                self.model_module = model_module
                self.loss_adapter = loss_adapter
                self.metric_accumulator = ObservableCameraRayFitV4MetricAccumulator
                self.derive_targets = derive_observable_camera_ray_evidence_v4_targets
                self.soft_rasterize = soft_rasterize_observable_camera_ray_evidence_v4
                self.file_records = {
                    str(item["path"]): item
                    for item in context.raw_manifest.get("files", [])
                    if isinstance(item, Mapping)
                }
                if not self.file_records:
                    raise ValueError("raw manifest file inventory is empty")
                self.payload_cache: dict[tuple[str, str, str], bytes] = {}

            def _payload_bytes(
                self,
                *,
                relative_path: str,
                expected_sha256: str,
                role: str,
                arm: str,
                stage: str,
                operation: str,
            ) -> bytes:
                key = (arm, stage, relative_path)
                cached = self.payload_cache.get(key)
                if cached is not None:
                    return cached
                raw = _read_bound_repo_input(
                    self.context.events,
                    relative_path=relative_path,
                    expected_sha256=expected_sha256,
                    stage=stage,
                    role=role,
                    arm=arm,
                    operation=operation,
                )
                self.payload_cache[key] = raw
                return raw

            def _clear_payload_cache(self, *, arm: str, stage: str) -> None:
                for key in [
                    item
                    for item in self.payload_cache
                    if item[0] == arm and item[1] == stage
                ]:
                    del self.payload_cache[key]

            def _validate_device(self) -> tuple[Any, dict[str, Any]]:
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise PermissionError("exact training requires one visible ROCm GPU")
                device = torch.device("cuda:0")
                properties = torch.cuda.get_device_properties(device)
                if (
                    str(properties.name) != policy.DEVICE_CONTRACT["device_name"]
                    or int(properties.total_memory)
                    < policy.DEVICE_CONTRACT["minimum_total_memory_bytes"]
                ):
                    raise PermissionError("exact training R9700 identity changed")
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                return device, {
                    "device": "cuda:0",
                    "name": str(properties.name),
                    "total_memory_bytes": int(properties.total_memory),
                    "visible_device_count": 1,
                    "torch_version": str(torch.__version__),
                    "hip_version": str(torch.version.hip),
                }

            def _read_raw_index(self, relative: str) -> bytes:
                record = self.file_records.get(relative)
                if not isinstance(record, Mapping):
                    raise PermissionError(f"raw file is absent from manifest: {relative}")
                full = f"{policy.RAW_SUPERVISION_ROOT_RELATIVE_PATH}/{relative}"
                return _read_bound_repo_input(
                    self.context.events,
                    relative_path=full,
                    expected_sha256=str(record["file_sha256"]),
                    stage="exact_input",
                    role="raw_supervision_manifest",
                    operation="read_bound_role_index",
                )

            def _load_indexes(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                pairs = _parse_jsonl(self._read_raw_index("pairs.jsonl"), name="pairs")
                endpoints = _parse_jsonl(
                    self._read_raw_index("endpoints.jsonl"),
                    name="endpoints",
                )
                for role in policy.DEVELOPMENT_ROLES:
                    if sum(item.get("dataset_role") == role for item in pairs) != policy.ROLE_COUNTS[role]["pairs"]:
                        raise PermissionError(f"raw pair count changed for {role}")
                    if sum(item.get("dataset_role") == role for item in endpoints) != policy.ROLE_COUNTS[role]["unique_endpoints"]:
                        raise PermissionError(f"raw endpoint count changed for {role}")
                return pairs, endpoints

            def _schedule(self, train_pairs: Sequence[Mapping[str, Any]]) -> tuple[list[int], dict[str, Any]]:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(policy.SCHEDULE_SEED)
                values: list[int] = []
                while len(values) < policy.PRESENTATION_COUNT:
                    values.extend(torch.randperm(policy.TRAIN_PAIR_COUNT, generator=generator).tolist())
                indices = values[: policy.PRESENTATION_COUNT]
                ordered_pair_ids = [str(item["content_sha256"]) for item in train_pairs]
                commitment = policy.schedule_commitment(
                    indices,
                    ordered_pair_ids,
                )
                commitment_core = dict(commitment)
                commitment_core.pop("content_sha256")
                presentations = [ordered_pair_ids[index] for index in indices]
                full = policy.content_value(
                    {
                        **commitment_core,
                        "ordered_train_pair_ids": ordered_pair_ids,
                        "presentation_indices": indices,
                        "per_update_pair_ids": [
                            presentations[offset : offset + policy.EFFECTIVE_BATCH_SIZE]
                            for offset in range(
                                0,
                                policy.PRESENTATION_COUNT,
                                policy.EFFECTIVE_BATCH_SIZE,
                            )
                        ],
                    }
                )
                return indices, full

            def _load_v4_and_initialize(self) -> tuple[dict[str, Any], bytes, str]:
                bindings = self.context.reservation["required_exact_bindings"]
                ladder_rows = policy.validate_camera_v14_ladder_rows(
                    self.context.manifest["camera_ladder_rows"],
                    reviewed_source_bindings=self.context.manifest[
                        "camera_v14_source_bindings"
                    ],
                )
                primary_row = ladder_rows[3]
                if (
                    primary_row["seed"] != 20260710
                    or primary_row["fit_size"] != 320
                    or primary_row["migratable"] is not True
                    or primary_row["rung_review"]["checkpoint_file_sha256"]
                    != primary_row["checkpoint"]["file_sha256"]
                    or primary_row["checkpoint"]["file_sha256"]
                    != bindings[
                        "v4_primary_seed_20260710_n320_checkpoint_file_sha256"
                    ]
                ):
                    raise PermissionError("sole Camera V14 migration row changed")
                checkpoint_raw = _read_bound_repo_input(
                    self.context.events,
                    relative_path=primary_row["checkpoint"]["path"],
                    expected_sha256=primary_row["checkpoint"]["file_sha256"],
                    stage="exact_input",
                    role="camera_v14_primary_checkpoint",
                    operation="open_primary_migration_checkpoint",
                )
                checkpoint = torch.load(
                    io.BytesIO(checkpoint_raw),
                    map_location="cpu",
                    weights_only=False,
                )
                if (
                    not isinstance(checkpoint, Mapping)
                    or checkpoint.get("model_class")
                    != "ObservableCameraRayEvidenceV4Model"
                    or not isinstance(checkpoint.get("state_dict"), Mapping)
                ):
                    raise PermissionError("primary V4 checkpoint contract changed")
                fit = ObservableCameraRayEvidenceV4Model()
                fit.load_state_dict(checkpoint["state_dict"], strict=True)
                torch.random.default_generator.manual_seed(
                    policy.INITIALIZATION_SEED
                )
                config = model_module.SharedObservableCameraRayJepaV5Config()
                shared = model_module.SharedObservableCameraRayJepaV5(config)
                migration = shared.migrate_from_fit_model(fit)
                state = {
                    name: value.detach().cpu().contiguous().clone()
                    for name, value in sorted(shared.state_dict().items())
                }
                state_sha256 = model_module.tensor_state_dict_sha256(state)
                buffer = io.BytesIO()
                torch.save(state, buffer)
                serialized = buffer.getvalue()
                receipt = {
                    "schema": policy.INITIALIZATION_SCHEMA,
                    "seed": policy.INITIALIZATION_SEED,
                    "primary_v4_seed": policy.PRIMARY_V4_SEED,
                    "primary_v4_fit_size": 320,
                    "migration": {
                        "fit_model_state_sha256": migration.fit_model_state_sha256,
                        "shared_encoder_state_sha256": migration.shared_encoder_state_sha256,
                        "evidence_head_state_sha256": migration.evidence_head_state_sha256,
                        "migrated_head_key_count": migration.migrated_head_key_count,
                        "source_shape": list(migration.source_shape),
                        "pixel_ray_shape": list(migration.pixel_ray_shape),
                    },
                    "hard_sync_count_before_training": 1,
                    "complete_training_state_sha256": state_sha256,
                    "serialized_training_state_file_sha256": hashlib.sha256(serialized).hexdigest(),
                    "arms": list(policy.ARMS),
                    "arm_initial_state_sha256": {
                        arm: state_sha256 for arm in policy.ARMS
                    },
                    "identical_before_optimizer_construction": True,
                    "device_of_initialization": "cpu",
                    "precision": "float32",
                }
                del fit, shared
                return receipt, serialized, state_sha256

            def _checkpoint_bytes(
                self,
                model: Any,
                optimizer: Any,
                *,
                arm: str,
                update: int,
                initialization_sha256: str,
                schedule_sha256: str,
            ) -> tuple[bytes, str, str]:
                state = {
                    name: value.detach().cpu().contiguous()
                    for name, value in sorted(model.state_dict().items())
                }
                state_sha = model_module.tensor_state_dict_sha256(state)
                semantic = {
                    "schema": "lewm_go2_shared_jepa_v5_full_training_v4_checkpoint_v1",
                    "arm": arm,
                    "update": update,
                    "model_config": model.model_config.to_dict(),
                    "model_state_sha256": state_sha,
                    "initialization_state_sha256": initialization_sha256,
                    "schedule_content_sha256": schedule_sha256,
                    "optimizer_contract": policy.OPTIMIZER_CONTRACT,
                    "development_only": True,
                    "runtime_ready": False,
                }
                content_sha = policy.canonical_json_sha256(semantic)
                buffer = io.BytesIO()
                torch.save(
                    {
                        **semantic,
                        "content_sha256": content_sha,
                        "model_state_dict": state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "cpu_rng_state": torch.get_rng_state(),
                        "gpu_rng_state": torch.cuda.get_rng_state(0),
                    },
                    buffer,
                )
                return buffer.getvalue(), content_sha, state_sha

            def _training_step_loss(self, model: Any, batch: Mapping[str, Any], arm: str) -> tuple[Any, Mapping[str, Any]]:
                pair = model.forward_training_pair(**batch["forward"])
                joint = self.loss_adapter.combine_joint_losses_v4(
                    model,
                    pair,
                    batch["current_supervision"],
                    batch["next_supervision"],
                )
                backward = (
                    joint.total
                    if arm == "promoted_jepa"
                    else joint.observable_camera_ray_v4.total
                )
                return backward, {
                    "joint": joint,
                    "pair": pair,
                }

            def _batch_loader(
                self,
                pairs: Sequence[Mapping[str, Any]],
                endpoints: Mapping[str, Mapping[str, Any]],
                indices: Sequence[int],
                primitive_vocabulary: Sequence[str],
                commanded_table: Any,
                device: Any,
                *,
                arm: str,
                stage: str,
            ) -> Mapping[str, Any]:
                selected = [pairs[index] for index in indices]
                current = [self._load_endpoint(endpoints[str(item["current_endpoint_sha256"])], device, arm=arm, stage=stage) for item in selected]
                next_frames = [self._load_endpoint(endpoints[str(item["next_endpoint_sha256"])], device, arm=arm, stage=stage) for item in selected]
                action_indices = [primitive_vocabulary.index(str(item["primitive"])) for item in selected]
                action = torch.zeros((len(selected), len(primitive_vocabulary)), dtype=torch.float32, device=device)
                action[torch.arange(len(selected), device=device), torch.tensor(action_indices, device=device)] = 1.0
                wrong_action = torch.roll(action, shifts=1, dims=1)
                realized = torch.tensor([item["relative_se2_current_frame"] for item in selected], dtype=torch.float32, device=device)
                commanded = action @ commanded_table
                wrong_commanded = wrong_action @ commanded_table
                stack = lambda frames, name: torch.stack([item[name] for item in frames], dim=0)
                current_supervision = self._supervision(current)
                next_supervision = self._supervision(next_frames)
                config = model_module.SharedObservableCameraRayJepaV5Config()
                return {
                    "forward": {
                        "current_image": stack(current, "image"),
                        "next_image": stack(next_frames, "image"),
                        "action": action,
                        "realized_delta_pose_current": realized,
                        "commanded_delta_pose_current": commanded,
                        "current_camera_origin_body_m": stack(current, "camera_origin"),
                        "current_camera_basis_body_fru": stack(current, "camera_basis"),
                        "current_ground_plane_z_body_m": stack(current, "ground"),
                        "next_camera_origin_body_m": stack(next_frames, "camera_origin"),
                        "next_camera_basis_body_fru": stack(next_frames, "camera_basis"),
                        "next_ground_plane_z_body_m": stack(next_frames, "ground"),
                        "next_prediction_mask": torch.ones((len(selected), *config.bev_size), dtype=torch.bool, device=device),
                        "diagnostic_wrong_action": wrong_action,
                        "diagnostic_wrong_action_delta_pose_current": wrong_commanded,
                        "diagnostic_wrong_commanded_delta_pose_current": -commanded,
                    },
                    "current_supervision": current_supervision,
                    "next_supervision": next_supervision,
                    "families": [str(item["family"]) for item in selected],
                }

            def _supervision(self, frames: Sequence[Mapping[str, Any]]) -> Any:
                stack = lambda name: torch.stack([item[name] for item in frames], dim=0)
                return model_module.ObservableCameraRayV4FrameSupervisionV5(
                    pixel_hit_mask=stack("pixel_hit").bool(),
                    pixel_first_hit_distance_m=stack("pixel_distance"),
                    ground_support_in_frustum=stack("ground_in_frustum").bool(),
                    ground_support_clear_to_target=stack("ground_clear").bool(),
                    target_raster_labels=stack("raster_labels").long(),
                )

            def _load_endpoint(self, endpoint: Mapping[str, Any], device: Any, *, arm: str, stage: str) -> dict[str, Any]:
                role = str(endpoint["dataset_role"])
                shard_path = str(endpoint["scene_shard"])
                shard_record = self.file_records.get(shard_path)
                if not isinstance(shard_record, Mapping):
                    raise PermissionError("endpoint shard is absent from manifest")
                shard_raw = self._payload_bytes(
                    relative_path=f"{policy.RAW_SUPERVISION_ROOT_RELATIVE_PATH}/{shard_path}",
                    expected_sha256=str(shard_record["file_sha256"]),
                    role=role,
                    arm=arm,
                    stage=stage,
                    operation="open_bound_scene_shard",
                )
                shard = _parse_content_json(shard_raw, name="scene shard")
                records = {str(item["path"]): item for item in shard["files"]}
                row = int(endpoint["shard_row"])
                def row_array(filename: str, dtype: str, shape: tuple[int, ...]) -> Any:
                    record = records[filename]
                    relative = f"{Path(shard_path).parent}/{filename}"
                    raw = self._payload_bytes(
                        relative_path=f"{policy.RAW_SUPERVISION_ROOT_RELATIVE_PATH}/{relative}",
                        expected_sha256=str(record["file_sha256"]),
                        role=role,
                        arm=arm,
                        stage=stage,
                        operation="open_bound_raw_supervision_array",
                    )
                    values = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(tuple(record["shape"]))
                    return torch.from_numpy(values[row].copy()).to(device=device)
                image_path = str(endpoint["image_path_metadata_only"])
                image_raw = self._payload_bytes(
                    relative_path=image_path,
                    expected_sha256=str(endpoint["image_sha256_commitment_only"]),
                    role=role,
                    arm=arm,
                    stage=stage,
                    operation="open_bound_rgb",
                )
                with Image.open(io.BytesIO(image_raw)) as decoded:
                    image_size = int(
                        model_module.SharedObservableCameraRayJepaV5Config().image_size
                    )
                    rgb = decoded.convert("RGB").resize(
                        (image_size, image_size),
                        Image.Resampling.BILINEAR,
                    )
                    if rgb.size != (image_size, image_size):
                        raise ValueError("exact resized RGB shape changed")
                    array = np.asarray(rgb, dtype=np.float32) / 255.0
                image = torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device)
                mean = image.new_tensor(model_module.NORMALIZATION_MEAN)[:, None, None]
                std = image.new_tensor(model_module.NORMALIZATION_STD)[:, None, None]
                return {
                    "image": (image - mean) / std,
                    "camera_origin": row_array("camera_origin_body_m.f4", "<f4", (3,)),
                    "camera_basis": row_array("camera_basis_body_fru.f4", "<f4", (3, 3)),
                    "ground": row_array("ground_plane_z_body_m.f4", "<f4", ()),
                    "ground_in_frustum": row_array("ground_support_in_frustum.u1", "u1", (128, 128, 5)),
                    "ground_clear": row_array("ground_support_clear_to_target.u1", "u1", (128, 128, 5)),
                    "pixel_hit": row_array("pixel_hit_mask.u1", "u1", (84, 112)),
                    "pixel_distance": row_array("pixel_first_hit_distance_m.f4", "<f4", (84, 112)),
                    "raster_labels": row_array("raster_labels.u1", "u1", (64, 64)),
                }

            def _train_arm(
                self,
                *,
                arm: str,
                initial_state: Mapping[str, Any],
                initial_state_sha256: str,
                schedule: Sequence[int],
                schedule_content_sha256: str,
                train_pairs: Sequence[Mapping[str, Any]],
                endpoints: Mapping[str, Mapping[str, Any]],
                primitive_vocabulary: Sequence[str],
                commanded_table: Any,
                device: Any,
            ) -> dict[int, bytes]:
                torch.manual_seed(policy.INITIALIZATION_SEED)
                torch.cuda.manual_seed_all(policy.INITIALIZATION_SEED)
                model = model_module.SharedObservableCameraRayJepaV5().to(device)
                model.load_state_dict(initial_state, strict=True)
                optimizer = torch.optim.AdamW(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    lr=policy.learning_rate(1),
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=1e-4,
                    amsgrad=False,
                )
                trace_rows: list[dict[str, Any]] = []
                checkpoint_raw: dict[int, bytes] = {}
                for update in range(1, policy.UPDATE_COUNT + 1):
                    learning_rate = policy.learning_rate(update)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    optimizer.zero_grad(set_to_none=True)
                    component_sums: dict[str, float] = {}
                    start = (update - 1) * policy.EFFECTIVE_BATCH_SIZE
                    update_indices = schedule[start : start + policy.EFFECTIVE_BATCH_SIZE]
                    for micro in range(policy.ACCUMULATION_STEPS):
                        low = micro * policy.MICROBATCH_SIZE
                        batch = self._batch_loader(
                            train_pairs,
                            endpoints,
                            update_indices[low : low + policy.MICROBATCH_SIZE],
                            primitive_vocabulary,
                            commanded_table,
                            device,
                            arm=arm,
                            stage="gradient",
                        )
                        backward, diagnostics = self._training_step_loss(model, batch, arm)
                        if not bool(torch.isfinite(backward).item()):
                            raise FloatingPointError("exact training loss is nonfinite")
                        (backward / policy.ACCUMULATION_STEPS).backward()
                        joint = diagnostics["joint"]
                        current_v4 = joint.observable_camera_ray_v4.current
                        next_v4 = joint.observable_camera_ray_v4.next
                        for name, value in {
                            "backward": backward,
                            "joint_total": joint.total,
                            "jepa_total": joint.established_jepa.total,
                            "jepa_prediction": joint.established_jepa.prediction,
                            "jepa_equivariance": joint.established_jepa.equivariance,
                            "jepa_action_contrast": joint.established_jepa.action_contrast,
                            "jepa_variance": joint.established_jepa.variance,
                            "jepa_warped_persistence": joint.established_jepa.warped_persistence,
                            "pair_v4_total": joint.observable_camera_ray_v4.total,
                            "current_v4_hierarchical_first_hit_nll": current_v4.hierarchical_first_hit_nll,
                            "current_v4_target_bin_offset_smooth_l1": current_v4.target_bin_offset_smooth_l1,
                            "current_v4_ground_clear_distance_state_balanced_bce": current_v4.ground_clear_distance_state_balanced_bce,
                            "current_v4_derived_raster_hierarchical_bce": current_v4.derived_raster_hierarchical_bce.total,
                            "current_v4_derived_raster_cell_nll": current_v4.derived_raster_cell_nll,
                            "next_v4_hierarchical_first_hit_nll": next_v4.hierarchical_first_hit_nll,
                            "next_v4_target_bin_offset_smooth_l1": next_v4.target_bin_offset_smooth_l1,
                            "next_v4_ground_clear_distance_state_balanced_bce": next_v4.ground_clear_distance_state_balanced_bce,
                            "next_v4_derived_raster_hierarchical_bce": next_v4.derived_raster_hierarchical_bce.total,
                            "next_v4_derived_raster_cell_nll": next_v4.derived_raster_cell_nll,
                        }.items():
                            component_sums[name] = component_sums.get(name, 0.0) + float(value.detach().cpu()) / policy.ACCUMULATION_STEPS
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not bool(torch.isfinite(gradient_norm).item()):
                        raise FloatingPointError("exact gradient norm is nonfinite")
                    gradient_after_clip = math.sqrt(
                        sum(
                            float(parameter.grad.detach().float().square().sum().cpu())
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        )
                    )
                    optimizer.step()
                    model.update_ema_target_after_optimizer_step()
                    trace_rows.append(
                        {
                            "schema": "lewm_go2_shared_jepa_v5_full_training_v4_trace_row_v1",
                            "arm": arm,
                            "update": update,
                            "learning_rate": learning_rate,
                            "microbatch_count": policy.ACCUMULATION_STEPS,
                            "optimizer_step_count": update,
                            "ema_step_count": update,
                            "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                            "gradient_norm_after_clip": gradient_after_clip,
                            "losses": component_sums,
                        }
                    )
                    if update in policy.CHECKPOINT_UPDATES:
                        raw, content_sha, state_sha = self._checkpoint_bytes(
                            model,
                            optimizer,
                            arm=arm,
                            update=update,
                            initialization_sha256=initial_state_sha256,
                            schedule_sha256=schedule_content_sha256,
                        )
                        relative = f"arms/{arm}/checkpoints/update_{update}.pt"
                        _publish_bytes(
                            self.context,
                            relative,
                            raw,
                            content_sha256=content_sha,
                            state_sha256=state_sha,
                        )
                        checkpoint_raw[update] = raw
                trace_raw = b"".join(
                    policy.canonical_json_bytes(row) + b"\n" for row in trace_rows
                )
                _publish_bytes(
                    self.context,
                    f"arms/{arm}/training_trace.jsonl",
                    trace_raw,
                    content_sha256=policy.canonical_json_sha256(trace_rows),
                )
                self._clear_payload_cache(arm=arm, stage="gradient")
                return checkpoint_raw

            def _evaluation_placeholder_from_raw_accumulators(
                self,
                checkpoint_raw: bytes,
                *,
                update: int,
                pairs: Sequence[Mapping[str, Any]],
                endpoints: Mapping[str, Mapping[str, Any]],
                primitive_vocabulary: Sequence[str],
                commanded_table: Any,
                device: Any,
                arm: str,
                stage: str,
            ) -> dict[str, Any]:
                """Run the fixed evaluation kernel and return raw scoped metrics.

                The same kernel is used at update zero and each frozen checkpoint;
                it never accepts caller-provided metrics or qualification state.
                """
                checkpoint = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=False)
                model = model_module.SharedObservableCameraRayJepaV5().to(device).eval()
                state = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state, strict=True)
                # Full inference is deliberately expressed through the reviewed
                # pair forward.  Raw sums and counts, not rounded means, feed the
                # pure frozen policy below.
                raw_scopes: dict[str, dict[str, Any]] = {}
                scoped_pairs = {
                    "aggregate": list(pairs),
                    **{family: [item for item in pairs if item["family"] == family] for family in policy.FAMILIES},
                }
                for scope, scope_pairs in scoped_pairs.items():
                    if not scope_pairs:
                        raise ValueError(f"selection scope is empty: {scope}")
                    physical = self._physical_scope_metrics(
                        model,
                        scope_pairs,
                        endpoints,
                        device,
                        arm=arm,
                        stage=stage,
                    )
                    jepa = self._jepa_scope_metrics(
                        model,
                        scope_pairs,
                        endpoints,
                        primitive_vocabulary,
                        commanded_table,
                        device,
                        arm=arm,
                        stage=stage,
                    )
                    raw_scopes[scope] = {"physical": physical, "jepa": jepa}
                aggregate = raw_scopes["aggregate"]
                return {
                    "update": update,
                    "scopes": raw_scopes,
                    "aggregate_complete_v4_loss": aggregate["physical"]["complete_v4_loss"],
                    "aggregate_prediction_to_persistence_ratio": aggregate["jepa"]["prediction_to_warped_persistence_ratio"],
                }

            def _physical_scope_metrics(self, model: Any, pairs: Sequence[Mapping[str, Any]], endpoints: Mapping[str, Mapping[str, Any]], device: Any, *, arm: str, stage: str) -> dict[str, Any]:
                unique_ids = sorted({str(item[f"{side}_endpoint_sha256"]) for item in pairs for side in ("current", "next")})
                frames = [self._load_endpoint(endpoints[digest], device, arm=arm, stage=stage) for digest in unique_ids]
                families = [str(endpoints[digest]["family"]) for digest in unique_ids]
                mapping_by_family: dict[str, list[int]] = {}
                for index, family in enumerate(families):
                    mapping_by_family.setdefault(family, []).append(index)
                wrong_mapping = list(range(len(frames)))
                for members in mapping_by_family.values():
                    for offset, index in enumerate(members):
                        wrong_mapping[index] = members[(offset + 1) % len(members)]
                correct_acc = self.metric_accumulator()
                wrong_acc = self.metric_accumulator()
                loss_sum = 0.0
                with torch.no_grad():
                    for start in range(0, len(frames), policy.MICROBATCH_SIZE):
                        indices = list(range(start, min(start + policy.MICROBATCH_SIZE, len(frames))))
                        target = [frames[index] for index in indices]
                        wrong = [frames[wrong_mapping[index]] for index in indices]
                        for source, accumulator in ((target, correct_acc), (wrong, wrong_acc)):
                            image = torch.stack([item["image"] for item in source])
                            origin = torch.stack([item["camera_origin"] for item in target])
                            basis = torch.stack([item["camera_basis"] for item in target])
                            ground = torch.stack([item["ground"] for item in target])
                            online = model.forward_frame(image, origin, basis, ground)
                            supervision = self._supervision(target)
                            targets = self.derive_targets(
                                pixel_hit_mask=supervision.pixel_hit_mask,
                                pixel_first_hit_distance_m=supervision.pixel_first_hit_distance_m,
                                ground_support_in_frustum=supervision.ground_support_in_frustum,
                                ground_support_clear_to_target=supervision.ground_support_clear_to_target,
                            )
                            soft = self.soft_rasterize(
                                online.evidence,
                                camera_origin_body_m=origin,
                                camera_basis_body_fru=basis,
                            )
                            accumulator.update(
                                raw_output=online.evidence,
                                targets=targets,
                                soft_raster=soft,
                                target_raster_labels=supervision.target_raster_labels,
                                families=[families[index] for index in indices],
                            )
                            if accumulator is correct_acc:
                                loss = self.loss_adapter.observable_camera_ray_v4_loss_v4(
                                    model,
                                    self._single_frame_pair_adapter(online),
                                    supervision,
                                    supervision,
                                    require_b4=False,
                                ).total
                                loss_sum += float(loss.cpu()) * len(indices)
                correct = correct_acc.finalize()
                wrong = wrong_acc.finalize()
                return self._flatten_physical_metrics(correct, wrong, loss_sum / len(frames))

            def _single_frame_pair_adapter(self, frame: Any) -> Any:
                return model_module.SharedTrainingPairV5(
                    current=frame,
                    next=frame,
                    predicted_next_bev=frame.bev,
                    stop_gradient_target_next_bev=frame.bev.detach(),
                    commanded_warped_current_bev=frame.bev,
                    commanded_overlap_mask=torch.ones_like(frame.bev[:, :1], dtype=torch.bool),
                    realized_warped_current_bev=frame.bev,
                    realized_overlap_mask=torch.ones_like(frame.bev[:, :1], dtype=torch.bool),
                    jepa=None,
                )

            def _flatten_physical_metrics(self, correct: Mapping[str, Any], wrong: Mapping[str, Any], complete_v4_loss: float) -> dict[str, Any]:
                c_depth = correct["pixel_hit_depth"]
                w_depth = wrong["pixel_hit_depth"]
                c_raster = correct["derived_raster"]
                w_raster = wrong["derived_raster"]
                distance = [value["balanced_accuracy"] for value in correct["ground_clear"]["by_distance_m"].values() if value["count"] > 0]
                recalls = {name: value for name, value in c_raster["class_recalls"].items() if value is not None}
                return {
                    "pixel_first_hit_balanced_accuracy": correct["pixel_hit_no_hit"]["balanced_accuracy"],
                    "depth_median_error_m": c_depth["median_absolute_error_m"],
                    "depth_p95_error_m": c_depth["p95_absolute_error_m"],
                    "ground_clear_balanced_accuracy": correct["ground_clear"]["overall"]["balanced_accuracy"],
                    "distance_group_balanced_accuracy": distance,
                    "derived_raster_nll": c_raster["nll"],
                    "derived_raster_balanced_accuracy": c_raster["balanced_accuracy"],
                    "present_class_recall": recalls,
                    "wrong_rgb_pixel_balanced_accuracy_drop": correct["pixel_hit_no_hit"]["balanced_accuracy"] - wrong["pixel_hit_no_hit"]["balanced_accuracy"],
                    "wrong_rgb_depth_median_error_increase_m": w_depth["median_absolute_error_m"] - c_depth["median_absolute_error_m"],
                    "wrong_rgb_depth_p95_error_increase_m": w_depth["p95_absolute_error_m"] - c_depth["p95_absolute_error_m"],
                    "wrong_rgb_ground_balanced_accuracy_drop": correct["ground_clear"]["overall"]["balanced_accuracy"] - wrong["ground_clear"]["overall"]["balanced_accuracy"],
                    "wrong_rgb_raster_nll_increase": w_raster["nll"] - c_raster["nll"],
                    "wrong_rgb_raster_balanced_accuracy_drop": c_raster["balanced_accuracy"] - w_raster["balanced_accuracy"],
                    "complete_v4_loss": complete_v4_loss,
                }

            def _jepa_scope_metrics(self, model: Any, pairs: Sequence[Mapping[str, Any]], endpoints: Mapping[str, Mapping[str, Any]], primitive_vocabulary: Sequence[str], commanded_table: Any, device: Any, *, arm: str, stage: str) -> dict[str, Any]:
                numerators: dict[str, float] = {}
                denominators: dict[str, int] = {}
                target_parts: list[Any] = []

                def normalized_error(prediction: Any, target: Any) -> Any:
                    return (
                        F.normalize(prediction, dim=1)
                        - F.normalize(target, dim=1)
                    ).square().mean(dim=1)

                def accumulate(name: str, values: Any, mask: Any) -> None:
                    if values.shape != mask.shape:
                        raise ValueError("raw JEPA accumulator shape changed")
                    weight = mask.to(values.dtype)
                    numerators[name] = numerators.get(name, 0.0) + float(
                        (values * weight).sum().cpu()
                    )
                    denominators[name] = denominators.get(name, 0) + int(
                        mask.sum().cpu()
                    )

                def mean(name: str) -> float:
                    return numerators.get(name, 0.0) / max(
                        1,
                        denominators.get(name, 0),
                    )

                with torch.no_grad():
                    for start in range(0, len(pairs), policy.MICROBATCH_SIZE):
                        batch_pairs = pairs[start : start + policy.MICROBATCH_SIZE]
                        batch = self._batch_loader(
                            pairs,
                            endpoints,
                            list(range(start, start + len(batch_pairs))),
                            primitive_vocabulary,
                            commanded_table,
                            device,
                            arm=arm,
                            stage=stage,
                        )
                        pair = model.forward_training_pair(**batch["forward"])
                        package = pair.jepa
                        scalar_checks = (
                            package.total,
                            package.prediction,
                            package.equivariance,
                            package.action_contrast,
                            package.variance,
                            package.warped_persistence,
                        )
                        if any(
                            not bool(torch.isfinite(value).item())
                            for value in scalar_checks
                        ):
                            raise FloatingPointError(
                                "JEPA package contains a nonfinite value"
                            )
                        target = pair.stop_gradient_target_next_bev.detach()
                        prediction = pair.predicted_next_bev.detach()
                        persistence = pair.commanded_warped_current_bev.detach()
                        prediction_mask = pair.commanded_overlap_mask[:, 0].bool()
                        prediction_error = normalized_error(prediction, target)
                        persistence_error = normalized_error(persistence, target)
                        accumulate("prediction", prediction_error, prediction_mask)
                        accumulate("persistence", persistence_error, prediction_mask)

                        current_bev = pair.current.bev.detach()
                        wrong_action_prediction, _wrong_warp, wrong_overlap = (
                            model.predict_from_command(
                                current_bev,
                                batch["forward"]["diagnostic_wrong_action"],
                                batch["forward"][
                                    "diagnostic_wrong_action_delta_pose_current"
                                ],
                            )
                        )
                        wrong_action_mask = prediction_mask & wrong_overlap[:, 0]
                        accumulate(
                            "wrong_action_real",
                            prediction_error,
                            wrong_action_mask,
                        )
                        accumulate(
                            "wrong_action_persistence",
                            persistence_error,
                            wrong_action_mask,
                        )
                        accumulate(
                            "wrong_action",
                            normalized_error(wrong_action_prediction, target),
                            wrong_action_mask,
                        )
                        accumulate(
                            "wrong_action_sensitivity",
                            normalized_error(wrong_action_prediction, prediction),
                            wrong_action_mask,
                        )

                        wrong_delta_prediction, _wrong_delta_warp, wrong_delta_overlap = (
                            model.predict_from_command(
                                current_bev,
                                batch["forward"]["action"],
                                batch["forward"][
                                    "diagnostic_wrong_commanded_delta_pose_current"
                                ],
                            )
                        )
                        wrong_delta_mask = prediction_mask & wrong_delta_overlap[:, 0]
                        accumulate(
                            "wrong_delta_real",
                            prediction_error,
                            wrong_delta_mask,
                        )
                        accumulate(
                            "wrong_delta_persistence",
                            persistence_error,
                            wrong_delta_mask,
                        )
                        accumulate(
                            "wrong_delta",
                            normalized_error(wrong_delta_prediction, target),
                            wrong_delta_mask,
                        )
                        accumulate(
                            "wrong_delta_sensitivity",
                            normalized_error(wrong_delta_prediction, prediction),
                            wrong_delta_mask,
                        )
                        target_parts.append(target.cpu())

                if not target_parts:
                    raise ValueError("raw JEPA target population is empty")
                target_float = torch.cat(target_parts, dim=0).float()
                if target_float.shape[0] < 2:
                    target_std = 0.0
                    target_rank = 0.0
                else:
                    target_std = float(
                        target_float.std(dim=0, unbiased=False).mean()
                    )
                    centered = target_float - target_float.mean(
                        dim=0,
                        keepdim=True,
                    )
                    samples = centered.permute(0, 2, 3, 1).reshape(
                        -1,
                        centered.shape[1],
                    )
                    if samples.shape[0] > 65_536:
                        stride = math.ceil(samples.shape[0] / 65_536)
                        samples = samples[::stride]
                    covariance = samples.T @ samples / max(
                        1,
                        samples.shape[0] - 1,
                    )
                    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
                    total = eigenvalues.sum()
                    if not bool((total > 0).item()):
                        target_rank = 0.0
                    else:
                        probabilities = eigenvalues / total
                        entropy = -(
                            probabilities
                            * probabilities.clamp_min(1e-12).log()
                        ).sum()
                        target_rank = float(torch.exp(entropy))

                prediction_mean = mean("prediction")
                persistence_mean = mean("persistence")
                wrong_action_advantage = mean("wrong_action") - mean(
                    "wrong_action_real"
                )
                wrong_delta_advantage = mean("wrong_delta") - mean(
                    "wrong_delta_real"
                )
                return {
                    "prediction_valid_cell_count": denominators.get(
                        "prediction",
                        0,
                    ),
                    "target_cross_sample_std_mean": target_std,
                    "target_cross_sample_effective_rank": target_rank,
                    "warped_persistence_target_change": persistence_mean,
                    "prediction_to_warped_persistence_ratio": (
                        prediction_mean / max(persistence_mean, 1e-8)
                    ),
                    "wrong_action_advantage_over_target_change": (
                        wrong_action_advantage
                        / max(mean("wrong_action_persistence"), 1e-8)
                    ),
                    "wrong_commanded_delta_advantage_over_target_change": (
                        wrong_delta_advantage
                        / max(mean("wrong_delta_persistence"), 1e-8)
                    ),
                    "wrong_action_prediction_sensitivity": mean(
                        "wrong_action_sensitivity"
                    ),
                    "wrong_commanded_delta_prediction_sensitivity": mean(
                        "wrong_delta_sensitivity"
                    ),
                }

            def _fit_calibration(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
                # Fixed six-parameter vector scaling.  The full logits/labels
                # collector is the same physical forward used above and runs on
                # all 759 unique calibration endpoints in input order.
                logits, labels, within_two_m, scope_masks = self._collect_calibration_logits(*_args, **_kwargs)
                logits = logits.to(device="cpu", dtype=torch.float32)
                labels = labels.to(device="cpu", dtype=torch.long)
                counts = torch.bincount(labels, minlength=3)
                if bool((counts == 0).any().item()):
                    raise ValueError("calibration role is missing a class")
                raw = torch.zeros(6, dtype=torch.float32, requires_grad=True)
                optimizer = torch.optim.LBFGS((raw,), lr=0.5, max_iter=80, line_search_fn="strong_wolfe")
                before = float(F.cross_entropy(logits, labels).item())
                if not math.isfinite(before):
                    raise FloatingPointError("uncalibrated NLL is nonfinite")
                def closure() -> Any:
                    optimizer.zero_grad(set_to_none=True)
                    scaled = self._scaled_logits(logits, raw)
                    loss = F.cross_entropy(scaled, labels)
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError("vector calibration NLL is nonfinite")
                    loss.backward()
                    return loss
                optimizer.step(closure)
                parameters = policy.centered_vector_scaling_parameters(
                    raw[:3].detach().tolist(),
                    raw[3:].detach().tolist(),
                )
                parameter_tensor = torch.tensor(parameters["log_scales"] + parameters["centered_biases"])
                calibrated_logits = self._scaled_logits(logits, parameter_tensor)
                after = float(F.cross_entropy(calibrated_logits, labels).item())
                if not math.isfinite(after) or after > before + 1e-6:
                    raise ValueError("calibration worsened NLL")
                probabilities = calibrated_logits.softmax(dim=1)
                grid_reports = self._threshold_reports(
                    probabilities,
                    labels,
                    within_two_m,
                    scope_masks["aggregate"],
                )
                threshold = policy.select_calibration_threshold(grid_reports)
                scope_reports = {}
                for scope, mask in scope_masks.items():
                    report = self._report_fixed_threshold(
                        probabilities[mask],
                        labels[mask],
                        within_two_m[mask],
                        threshold,
                    )
                    report["uncalibrated_nll"] = float(
                        F.cross_entropy(logits[mask], labels[mask]).item()
                    )
                    report["calibrated_nll"] = float(
                        F.cross_entropy(
                            calibrated_logits[mask],
                            labels[mask],
                        ).item()
                    )
                    report["class_counts"] = torch.bincount(
                        labels[mask],
                        minlength=3,
                    ).tolist()
                    if not math.isfinite(report["uncalibrated_nll"]) or not math.isfinite(
                        report["calibrated_nll"]
                    ):
                        raise FloatingPointError(
                            "calibration scope NLL is nonfinite: " + scope
                        )
                    scope_reports[scope] = report
                return {
                    "parameters": parameters,
                    "uncalibrated_nll": before,
                    "calibrated_nll": after,
                    "class_counts": counts.tolist(),
                    "threshold": threshold,
                    "scope_reports": scope_reports,
                }

            def _scaled_logits(self, logits: Any, raw: Any) -> Any:
                log_scales = raw[:3].clamp(-3.0, 3.0)
                biases = raw[3:] - raw[3:].mean()
                return logits * log_scales.exp()[None] + biases[None]

            def _collect_calibration_logits(self, model: Any, pairs: Sequence[Mapping[str, Any]], endpoints: Mapping[str, Mapping[str, Any]], device: Any, *, arm: str) -> tuple[Any, Any, Any, dict[str, Any]]:
                ids = sorted({str(item[f"{side}_endpoint_sha256"]) for item in pairs for side in ("current", "next")})
                all_logits = []
                all_labels = []
                within_two_parts = []
                family_ranges: dict[str, list[int]] = {family: [] for family in policy.FAMILIES}
                config = model.model_config
                rows, columns = config.bev_size
                forward_step = (config.forward_range_m[1] - config.forward_range_m[0]) / rows
                left_step = (config.left_range_m[1] - config.left_range_m[0]) / columns
                forward = config.forward_range_m[0] + (torch.arange(rows, dtype=torch.float32) + 0.5) * forward_step
                left = config.left_range_m[0] + (torch.arange(columns, dtype=torch.float32) + 0.5) * left_step
                grid_forward, grid_left = torch.meshgrid(forward, left, indexing="ij")
                frame_within_two = (grid_forward.square() + grid_left.square()).sqrt().reshape(-1) <= 2.0
                with torch.no_grad():
                    for digest in ids:
                        frame = self._load_endpoint(endpoints[digest], device, arm=arm, stage="calibration")
                        online = model.forward_frame(frame["image"][None], frame["camera_origin"][None], frame["camera_basis"][None], frame["ground"][None])
                        soft = self.soft_rasterize(online.evidence, camera_origin_body_m=frame["camera_origin"][None], camera_basis_body_fru=frame["camera_basis"][None])
                        logits = soft.class_probabilities.clamp_min(torch.finfo(torch.float32).eps).log().permute(0, 2, 3, 1).reshape(-1, 3).cpu()
                        labels = frame["raster_labels"].reshape(-1).cpu().long()
                        base = sum(item.shape[0] for item in all_labels)
                        family_ranges[str(endpoints[digest]["family"])].extend(range(base, base + labels.numel()))
                        all_logits.append(logits)
                        all_labels.append(labels)
                        within_two_parts.append(frame_within_two)
                logits = torch.cat(all_logits)
                labels = torch.cat(all_labels)
                within_two = torch.cat(within_two_parts)
                masks = {"aggregate": torch.ones(labels.numel(), dtype=torch.bool)}
                for family, indices in family_ranges.items():
                    mask = torch.zeros(labels.numel(), dtype=torch.bool)
                    mask[indices] = True
                    masks[family] = mask
                return logits, labels, within_two, masks

            def _threshold_reports(self, probabilities: Any, labels: Any, within_two_m: Any, mask: Any) -> dict[str, Any]:
                reports = {}
                for values in policy.threshold_grid():
                    key = policy.canonical_json_sha256(list(values))
                    reports[key] = self._threshold_counts(
                        probabilities[mask],
                        labels[mask],
                        within_two_m[mask],
                        values,
                    )
                return reports

            def _threshold_counts(self, probabilities: Any, labels: Any, within_two_m: Any, values: Sequence[float]) -> dict[str, int]:
                free_min, occupied_max, unknown_max, detection_min = values
                admitted = (probabilities[:, 1] >= free_min) & (probabilities[:, 2] <= occupied_max) & (probabilities[:, 0] <= unknown_max)
                free = labels == 1
                obstacle = (labels == 2) & within_two_m
                detected = probabilities[:, 2] >= detection_min
                return {
                    "admitted_free_count": int(admitted.sum()),
                    "admitted_free_true_free_count": int((admitted & free).sum()),
                    "useful_free_count": int(free.sum()),
                    "useful_free_admitted_count": int((free & admitted).sum()),
                    "obstacle_within_2m_count": int(obstacle.sum()),
                    "obstacle_within_2m_excluded_count": int((obstacle & ~admitted).sum()),
                    "obstacle_within_2m_detected_count": int((obstacle & detected).sum()),
                }

            def _report_fixed_threshold(self, probabilities: Any, labels: Any, within_two_m: Any, threshold: Mapping[str, Any]) -> dict[str, Any]:
                counts = self._threshold_counts(
                    probabilities,
                    labels,
                    within_two_m,
                    (
                        threshold["free_probability_minimum"],
                        threshold["occupied_probability_maximum"],
                        threshold["unknown_probability_maximum"],
                        threshold["occupied_detection_minimum"],
                    ),
                )
                admitted = counts["admitted_free_count"]
                useful = counts["useful_free_count"]
                obstacles = counts["obstacle_within_2m_count"]
                return {
                    **counts,
                    "admitted_free_precision": counts["admitted_free_true_free_count"] / admitted if admitted else None,
                    "useful_free_recall": counts["useful_free_admitted_count"] / useful if useful else None,
                    "obstacle_exclusion_recall_within_2m": counts["obstacle_within_2m_excluded_count"] / obstacles if obstacles else None,
                    "obstacle_detection_recall_within_2m": counts["obstacle_within_2m_detected_count"] / obstacles if obstacles else None,
                }

            def _numeric_metric_delta(self, promoted: Any, matched: Any) -> Any:
                if isinstance(promoted, Mapping) and isinstance(matched, Mapping):
                    if set(promoted) != set(matched):
                        raise ValueError("ablation metric fields changed")
                    return {
                        name: self._numeric_metric_delta(promoted[name], matched[name])
                        for name in promoted
                    }
                if (
                    isinstance(promoted, Sequence)
                    and not isinstance(promoted, (str, bytes))
                    and isinstance(matched, Sequence)
                    and not isinstance(matched, (str, bytes))
                ):
                    if len(promoted) != len(matched):
                        raise ValueError("ablation metric sequence changed")
                    return [
                        self._numeric_metric_delta(left, right)
                        for left, right in zip(promoted, matched, strict=True)
                    ]
                if (
                    isinstance(promoted, bool)
                    or isinstance(matched, bool)
                    or not isinstance(promoted, (int, float))
                    or not isinstance(matched, (int, float))
                ):
                    raise TypeError("ablation metric leaf is not numeric")
                delta = float(promoted) - float(matched)
                if not math.isfinite(delta):
                    raise FloatingPointError("ablation metric delta is nonfinite")
                return delta

            def _pre_g2_candidate_checkpoint(self, checkpoint_raw: bytes, selection: Mapping[str, Any], calibration: Mapping[str, Any]) -> tuple[bytes, str, str]:
                checkpoint = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=False)
                model = model_module.SharedObservableCameraRayJepaV5()
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                deployment = model.deployment_state_dict()
                deployment_sha = model_module.tensor_state_dict_sha256(deployment)
                core = policy.pre_g2_candidate_checkpoint_core(
                    model_config=model.model_config.to_dict(),
                    deployment_state_sha256=deployment_sha,
                    selection=selection,
                    calibration=calibration,
                )
                content_sha = policy.canonical_json_sha256(core)
                buffer = io.BytesIO()
                torch.save({**core, "content_sha256": content_sha, "deployment_state_dict": deployment}, buffer)
                return buffer.getvalue(), content_sha, deployment_sha

            def _completion_rehash(self) -> None:
                opened = list(self.context.events)
                seen: set[tuple[Any, ...]] = set()
                for event in opened:
                    identity = (
                        event["arm"],
                        event["role"],
                        event["relative_path"],
                        event["expected_sha256"],
                        event["byte_count"],
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    raw = _read_bound_repo_input(
                        self.context.events,
                        relative_path=str(event["relative_path"]),
                        expected_sha256=str(event["expected_sha256"]),
                        stage="completion_rehash",
                        role=str(event["role"]),
                        arm=event["arm"],
                        operation="rehash_before_completion",
                    )
                    if len(raw) != event["byte_count"]:
                        raise PermissionError("completion rehash byte count changed")

            def run(self) -> dict[str, Any]:
                initialization, serialized_initial, initial_sha = (
                    self._load_v4_and_initialize()
                )
                device, device_record = self._validate_device()
                pairs, endpoint_rows = self._load_indexes()
                endpoints = {str(item["endpoint_identity_sha256"]): item for item in endpoint_rows}
                train_pairs = [item for item in pairs if item["dataset_role"] == "train"]
                selection_pairs = [item for item in pairs if item["dataset_role"] == "checkpoint_selection"]
                calibration_pairs = [item for item in pairs if item["dataset_role"] == "probability_calibration"]
                schedule, schedule_artifact = self._schedule(train_pairs)
                _publish_json(self.context, "schedule.json", dict(schedule_artifact))
                primitive_vocabulary = sorted({str(item["primitive"]) for item in train_pairs})
                if len(primitive_vocabulary) != 9:
                    raise PermissionError("train primitive vocabulary changed")
                commanded_values = []
                for primitive in primitive_vocabulary:
                    rows = torch.tensor([item["relative_se2_current_frame"] for item in train_pairs if item["primitive"] == primitive], dtype=torch.float32)
                    commanded_values.append(torch.quantile(rows, 0.5, dim=0))
                commanded_table_cpu = torch.stack(commanded_values)
                commanded_table_value = commanded_table_cpu.tolist()
                initialization["primitive_vocabulary"] = primitive_vocabulary
                initialization["commanded_delta_table"] = commanded_table_value
                initialization["commanded_delta_table_sha256"] = policy.canonical_json_sha256(commanded_table_value)
                initialization_artifact = _publish_json(self.context, "initialization.json", initialization)
                initial_state = torch.load(io.BytesIO(serialized_initial), map_location="cpu", weights_only=True)
                commanded_table = commanded_table_cpu.to(device)
                promoted_checkpoints = self._train_arm(
                    arm="promoted_jepa",
                    initial_state=initial_state,
                    initial_state_sha256=initial_sha,
                    schedule=schedule,
                    schedule_content_sha256=schedule_artifact["content_sha256"],
                    train_pairs=train_pairs,
                    endpoints=endpoints,
                    primitive_vocabulary=primitive_vocabulary,
                    commanded_table=commanded_table,
                    device=device,
                )
                migration_baseline = self._evaluation_placeholder_from_raw_accumulators(
                    serialized_initial,
                    update=0,
                    pairs=selection_pairs,
                    endpoints=endpoints,
                    primitive_vocabulary=primitive_vocabulary,
                    commanded_table=commanded_table,
                    device=device,
                    arm="promoted_jepa",
                    stage="selection",
                )
                candidates = [
                    self._evaluation_placeholder_from_raw_accumulators(
                        promoted_checkpoints[update],
                        update=update,
                        pairs=selection_pairs,
                        endpoints=endpoints,
                        primitive_vocabulary=primitive_vocabulary,
                        commanded_table=commanded_table,
                        device=device,
                        arm="promoted_jepa",
                        stage="selection",
                    )
                    for update in policy.CHECKPOINT_UPDATES
                ]
                self._clear_payload_cache(
                    arm="promoted_jepa",
                    stage="selection",
                )
                checkpoint_metrics = _publish_json(
                    self.context,
                    "arms/promoted_jepa/checkpoint_metrics.json",
                    {
                        "schema": "lewm_go2_shared_jepa_v5_full_training_v4_checkpoint_metrics_v1",
                        "role": "checkpoint_selection",
                        "pair_count": 495,
                        "unique_endpoint_count": 924,
                        "migration_baseline_nonselectable": migration_baseline,
                        "candidates": candidates,
                    },
                )
                selection = policy.select_promoted_checkpoint(candidates)
                selection_artifact = _publish_json(
                    self.context,
                    "selection.json",
                    {
                        "schema": policy.SELECTION_SCHEMA,
                        **selection,
                        "checkpoint_metrics_content_sha256": checkpoint_metrics["content_sha256"],
                        "ablation_influenced_selection": False,
                        "calibration_influenced_selection": False,
                    },
                )
                selected_update = int(selection["selected_update"])
                matched_checkpoints = self._train_arm(
                    arm="matched_no_jepa",
                    initial_state=initial_state,
                    initial_state_sha256=initial_sha,
                    schedule=schedule,
                    schedule_content_sha256=schedule_artifact["content_sha256"],
                    train_pairs=train_pairs,
                    endpoints=endpoints,
                    primitive_vocabulary=primitive_vocabulary,
                    commanded_table=commanded_table,
                    device=device,
                )
                matched_metrics = self._evaluation_placeholder_from_raw_accumulators(
                    matched_checkpoints[selected_update],
                    update=selected_update,
                    pairs=selection_pairs,
                    endpoints=endpoints,
                    primitive_vocabulary=primitive_vocabulary,
                    commanded_table=commanded_table,
                    device=device,
                    arm="matched_no_jepa",
                    stage="diagnostic",
                )
                self._clear_payload_cache(
                    arm="matched_no_jepa",
                    stage="diagnostic",
                )
                _publish_json(
                    self.context,
                    "arms/matched_no_jepa/matched_update_metrics.json",
                    {
                        "schema": "lewm_go2_shared_jepa_v5_full_training_v4_matched_metrics_v1",
                        "selected_promoted_update": selected_update,
                        "metrics": matched_metrics,
                        "selection_effect": "none",
                    },
                )
                scene_id_by_family = {}
                for family in policy.FAMILIES:
                    scene_ids = {
                        str(item["scene_id"])
                        for item in selection_pairs
                        if item["family"] == family
                    }
                    if len(scene_ids) != 1:
                        raise PermissionError(
                            "selection family is not exactly one scene: " + family
                        )
                    scene_id_by_family[family] = next(iter(scene_ids))
                promoted_selected_metrics = candidates[
                    policy.CHECKPOINT_UPDATES.index(selected_update)
                ]
                diagnostic = _publish_json(
                    self.context,
                    "selection_role_ablation_diagnostic.json",
                    {
                        "schema": policy.DIAGNOSTIC_ABLATION_SCHEMA,
                        **policy.selection_role_ablation_contract(),
                        "scene_id_by_family": scene_id_by_family,
                        "promoted": promoted_selected_metrics,
                        "matched_no_jepa": matched_metrics,
                        "raw_delta_direction": "promoted_minus_matched_no_jepa",
                        "raw_metric_deltas": self._numeric_metric_delta(
                            promoted_selected_metrics["scopes"],
                            matched_metrics["scopes"],
                        ),
                    },
                )
                calibrations = {}
                for arm, raw in (
                    ("promoted_jepa", promoted_checkpoints[selected_update]),
                    ("matched_no_jepa", matched_checkpoints[selected_update]),
                ):
                    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
                    model = model_module.SharedObservableCameraRayJepaV5().to(device).eval()
                    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    calibration = self._fit_calibration(
                        model,
                        calibration_pairs,
                        endpoints,
                        device,
                        arm=arm,
                    )
                    self._clear_payload_cache(arm=arm, stage="calibration")
                    calibrations[arm] = _publish_json(
                        self.context,
                        f"calibration/{arm}.json",
                        {
                            "schema": policy.CALIBRATION_SCHEMA,
                            "arm": arm,
                            "role": "probability_calibration",
                            "pair_count": 415,
                            "unique_endpoint_count": 759,
                            **calibration,
                        },
                    )
                promoted_calibration = calibrations["promoted_jepa"]
                for scope, report in promoted_calibration["scope_reports"].items():
                    if (
                        len(report["class_counts"]) != 3
                        or any(count <= 0 for count in report["class_counts"])
                        or report["admitted_free_precision"] is None
                        or report["admitted_free_precision"] < 0.99
                        or report["useful_free_recall"] is None
                        or report["useful_free_recall"] < 0.90
                        or report["obstacle_exclusion_recall_within_2m"] is None
                        or report["obstacle_exclusion_recall_within_2m"] < 0.95
                        or report["obstacle_detection_recall_within_2m"] is None
                        or report["obstacle_detection_recall_within_2m"] < 0.95
                        or report["calibrated_nll"]
                        > report["uncalibrated_nll"] + 1e-6
                    ):
                        raise ValueError(f"calibrated development gate failed: {scope}")
                candidate_raw, candidate_content, candidate_state = self._pre_g2_candidate_checkpoint(
                    promoted_checkpoints[selected_update],
                    selection_artifact,
                    promoted_calibration,
                )
                _publish_bytes(
                    self.context,
                    "pre_g2_candidate_checkpoint.pt",
                    candidate_raw,
                    content_sha256=candidate_content,
                    state_sha256=candidate_state,
                )
                self._completion_rehash()
                ledger_summary = policy.validate_access_ledger(
                    self.context.events,
                    require_completion_rehash=True,
                )
                ledger = _publish_json(
                    self.context,
                    "access_ledger.json",
                    {
                        "schema": policy.ACCESS_LEDGER_SCHEMA,
                        "events": self.context.events,
                        "summary": ledger_summary,
                        "forbidden_open_count": 0,
                        "g2_open_count": 0,
                        "heldout_open_count": 0,
                        "runtime_navigation_hardware_open_count": 0,
                        "production_or_promotion_open_count": 0,
                    },
                )
                training_record = _publish_json(
                    self.context,
                    "training_record.json",
                    {
                        "schema": policy.TRAINING_RECORD_SCHEMA,
                        "status": "pre_g2_development_candidate_pending_independent_verification",
                        "device": device_record,
                        "initialization_content_sha256": initialization_artifact["content_sha256"],
                        "schedule_content_sha256": schedule_artifact["content_sha256"],
                        "selection_content_sha256": selection_artifact["content_sha256"],
                        "calibration_content_sha256": {arm: value["content_sha256"] for arm, value in calibrations.items()},
                        "diagnostic_content_sha256": diagnostic["content_sha256"],
                        "access_ledger_content_sha256": ledger["content_sha256"],
                        "optimizer_contract": policy.OPTIMIZER_CONTRACT,
                        "joint_loss_contract": policy.JOINT_LOSS_CONTRACT,
                        "runtime_ready": False,
                        "g2_authorized": False,
                        "heldout_authorized": False,
                        "production_or_promotion_authorized": False,
                        "retry_authorized": False,
                    },
                )
                return {
                    "training_record": training_record,
                    "selected_update": selected_update,
                    "pre_g2_candidate_state_sha256": candidate_state,
                }

        return FixedProductionTrainingBackend


    def _build_context(
        claim_fd: int,
        parent_fd: int,
        directory_name: str,
        manifest_file_sha256: str,
        manifest_content_sha256: str,
    ) -> TrainerContext:
        reservation, reservation_raw, identity = _load_reservation(
            claim_fd,
            parent_fd,
            directory_name,
            manifest_file_sha256,
            manifest_content_sha256,
        )
        events: list[dict[str, Any]] = []
        preflight, preflight_raw = _load_preflight_first(reservation, events)
        authority = _load_authority_after_preflight(
            reservation,
            manifest_file_sha256,
            manifest_content_sha256,
            events,
        )
        source_review = {
            "schema": policy.SOURCE_REVIEW_SCHEMA,
            "reviewed_sources": authority["source_hashes"],
            "camera_v14_dynamic_authority_bindings": {
                policy.CAMERA_V14_SOURCE_REVIEW_RELATIVE_PATH: reservation[
                    "required_exact_bindings"
                ]["camera_v14_source_review_file_sha256"],
                policy.CAMERA_V14_LADDER_PREREGISTRATION_RELATIVE_PATH: (
                    reservation["required_exact_bindings"]
                    ["camera_v14_ladder_preregistration_file_sha256"]
                ),
                policy.CAMERA_V14_LADDER_REVIEW_RELATIVE_PATH: reservation[
                    "required_exact_bindings"
                ]["camera_v14_ladder_independent_review_file_sha256"],
            },
            "implementation_review_file_sha256": reservation[
                "required_exact_bindings"
            ]["implementation_independent_review_file_sha256"],
            "frozen_parent_closure": policy.reviewed_source_bindings(),
            "live_navigation_readiness_hash_authoritative": False,
        }
        input_bindings = {
            "schema": policy.INPUT_BINDINGS_SCHEMA,
            "final_exact_authorization_file_sha256": manifest_file_sha256,
            "final_exact_authorization_content_sha256": manifest_content_sha256,
            "preflight_receipt_file_sha256": reservation["required_exact_bindings"][
                "preflight_receipt_file_sha256"
            ],
            "raw_manifest_file_sha256": reservation["required_exact_bindings"][
                "development_raw_supervision_manifest_file_sha256"
            ],
            "raw_manifest_content_sha256": authority["raw_manifest"]["content_sha256"],
            "raw_audit_file_sha256": reservation["required_exact_bindings"][
                "development_raw_supervision_audit_file_sha256"
            ],
            "raw_audit_content_sha256": authority["raw_audit"]["content_sha256"],
            "raw_v13_source_chain": policy.RAW_CHAIN_SOURCE_BINDINGS,
            "raw_v13_dataset_use_grant": policy.RAW_DATASET_USE_GRANT,
            "camera_v14_source_review_file_sha256": reservation[
                "required_exact_bindings"
            ]["camera_v14_source_review_file_sha256"],
            "camera_v14_source_review_content_sha256": authority[
                "camera_source_review"
            ]["content_sha256"],
            "camera_v14_n5_gate_file_sha256": reservation[
                "required_exact_bindings"
            ]["camera_v14_n5_gate_pass_file_sha256"],
            "camera_v14_n5_gate_content_sha256": authority["camera_n5_gate"][
                "content_sha256"
            ],
            "camera_v14_ladder_preregistration_file_sha256": reservation[
                "required_exact_bindings"
            ]["camera_v14_ladder_preregistration_file_sha256"],
            "camera_v14_ladder_independent_review_file_sha256": reservation[
                "required_exact_bindings"
            ]["camera_v14_ladder_independent_review_file_sha256"],
            "camera_v14_two_seed_ladder_file_sha256": reservation["required_exact_bindings"][
                "camera_v14_two_seed_ladder_pass_file_sha256"
            ],
            "camera_v14_two_seed_ladder_content_sha256": authority["v4_ladder"]["content_sha256"],
            "v4_primary_seed": 20260710,
            "v4_replication_seed": 20260711,
            "v4_primary_fit_size": 320,
            "camera_ladder_existing_attempt_count": 1,
            "camera_ladder_future_attempt_count": 7,
            "camera_ladder_aggregate_rung_count": 8,
            "seed_20260710_n5_reexecuted": False,
            "warm_start_used": False,
            "g2_authorized": False,
            "heldout_authorized": False,
            "runtime_navigation_hardware_authorized": False,
            "production_or_promotion_authorized": False,
        }
        return TrainerContext(
            claim_fd=claim_fd,
            parent_fd=parent_fd,
            directory_name=directory_name,
            directory_identity=identity,
            reservation=reservation,
            reservation_raw=reservation_raw,
            manifest=authority["manifest"],
            manifest_raw=authority["manifest_raw"],
            preflight_receipt=preflight,
            preflight_receipt_raw=preflight_raw,
            raw_manifest=authority["raw_manifest"],
            raw_manifest_raw=authority["raw_manifest_raw"],
            raw_audit=authority["raw_audit"],
            raw_audit_raw=authority["raw_audit_raw"],
            v4_ladder=authority["v4_ladder"],
            v4_ladder_raw=authority["v4_ladder_raw"],
            source_review=source_review,
            input_bindings=input_bindings,
            events=events,
        )


    def _finalize_summary(context: TrainerContext, result: Mapping[str, Any]) -> dict[str, Any]:
        reservation_binding = policy.artifact_binding(
            "reservation.json",
            context.reservation_raw,
            content_sha256=str(context.reservation["content_sha256"]),
        )
        context.published["reservation.json"] = reservation_binding
        expected = list(policy.EXACT_INVENTORY[:-1])
        if set(context.published) != set(expected):
            missing = sorted(set(expected) - set(context.published))
            extra = sorted(set(context.published) - set(expected))
            raise PermissionError(f"trainer inventory changed; missing={missing}, extra={extra}")
        ordered = {name: context.published[name] for name in expected}
        core = {
            "schema": "lewm_go2_shared_jepa_v5_full_training_v4_trainer_summary_v1",
            "status": "trainer_publication_complete",
            "directory_identity": list(context.directory_identity),
            "final_exact_authorization_content_sha256": context.manifest["content_sha256"],
            "artifacts_before_completion": ordered,
            "artifacts_before_completion_sha256": policy.canonical_json_sha256(ordered),
            "selected_update": result["selected_update"],
            "pre_g2_candidate_state_sha256": result[
                "pre_g2_candidate_state_sha256"
            ],
            "independent_verification_required": True,
            "runtime_ready": False,
            "g2_authorized": False,
            "heldout_authorized": False,
            "production_or_promotion_authorized": False,
        }
        return policy.content_value(core)


    def _orchestrate_trainer(
        reservation_loader: Callable[[], Any],
        preflight_first_loader: Callable[[Any], Any],
        authority_loader: Callable[[Any, Any], Any],
        backend_loader: Callable[[], Any],
        backend_runner: Callable[[Any, Any], Any],
    ) -> Any:
        reservation = reservation_loader()
        preflight = preflight_first_loader(reservation)
        authority = authority_loader(reservation, preflight)
        backend = backend_loader()
        return backend_runner(backend, authority)


    def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--claim-fd", required=True, type=int)
        parser.add_argument("--parent-fd", required=True, type=int)
        parser.add_argument("--expected-directory-name", required=True)
        parser.add_argument("--final-exact-authorization-sha256", required=True)
        parser.add_argument("--final-exact-authorization-content-sha256", required=True)
        parser.add_argument("--expected-source-sha256", required=True)
        args = parser.parse_args(argv)
        if (
            args.claim_fd < 0
            or args.parent_fd < 0
            or args.expected_directory_name != policy.CANONICAL_EXACT_ROOT.name
            or any(
                not policy.is_sha256(value)
                for value in (
                    args.final_exact_authorization_sha256,
                    args.final_exact_authorization_content_sha256,
                    args.expected_source_sha256,
                )
            )
        ):
            raise ValueError("exact trainer arguments are malformed")
        return args


    arguments = parse_args()
    context = _build_context(
        arguments.claim_fd,
        arguments.parent_fd,
        arguments.expected_directory_name,
        arguments.final_exact_authorization_sha256,
        arguments.final_exact_authorization_content_sha256,
    )
    if context.reservation["required_exact_bindings"]["exact_trainer_source_sha256"] != arguments.expected_source_sha256:
        raise PermissionError("exact trainer source binding changed")
    _publish_bootstrap_artifacts(context)
    backend_class = _fixed_production_backend_loader()
    result = backend_class(context).run()
    summary = _finalize_summary(context, result)
    print(policy.canonical_json_bytes(summary).decode("ascii"))
