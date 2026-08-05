#!/usr/bin/env python3
"""Independent reconstruction verifier for Shared JEPA V5 training V3.

This process does not import the trainer.  After validating the retained exact
reservation it opens the bound preflight receipt first, then independently
reopens sources, raw role inputs, V4 migration state, both arms' checkpoints,
selection/calibration payload, and every published artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lewm.benchmarks import go2_shared_jepa_v5_full_training_v3_policy as policy


if __name__ == "__main__":
    ROOT = SCRIPT_ROOT

    def _directory_flags() -> int:
        if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
            raise PermissionError("exact verifier requires no-follow directories")
        return (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )


    def _file_flags() -> int:
        if not getattr(os, "O_NOFOLLOW", 0):
            raise PermissionError("exact verifier requires no-follow files")
        return (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )


    def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
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


    def _read_fd(descriptor: int, *, name: str) -> bytes:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PermissionError(f"{name} is not singly linked")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _fingerprint(before) != _fingerprint(after):
            raise RuntimeError(f"{name} changed while independently verified")
        return b"".join(chunks)


    def _read_relative(base_fd: int, relative: str) -> bytes:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise PermissionError("independent verifier path escaped")
        descriptors: list[int] = []
        parent_fd = base_fd
        try:
            for component in path.parts[:-1]:
                descriptor = os.open(component, _directory_flags(), dir_fd=parent_fd)
                descriptors.append(descriptor)
                parent_fd = descriptor
            descriptor = os.open(path.name, _file_flags(), dir_fd=parent_fd)
            try:
                return _read_fd(descriptor, name=relative)
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


    def _read_repo(relative: str) -> bytes:
        root_fd = os.open(ROOT, _directory_flags())
        try:
            return _read_relative(root_fd, relative)
        finally:
            os.close(root_fd)


    def _assert_hash(raw: bytes, expected: str, *, name: str) -> None:
        if not policy.is_sha256(expected) or hashlib.sha256(raw).hexdigest() != expected:
            raise PermissionError(f"{name} file hash changed")


    def _parse_json(raw: bytes, *, name: str) -> dict[str, Any]:
        return policy.parse_canonical_json(raw, name=name)


    def _parse_jsonl(raw: bytes, *, name: str) -> list[dict[str, Any]]:
        if not raw or not raw.endswith(b"\n") or b"\n\n" in raw:
            raise ValueError(f"{name} is not canonical JSONL")
        rows = []
        for index, line in enumerate(raw.splitlines(), start=1):
            value = json.loads(line.decode("ascii"))
            if not isinstance(value, dict) or policy.canonical_json_bytes(value) != line:
                raise ValueError(f"{name} row {index} is noncanonical")
            core = dict(value)
            declared = core.pop("content_sha256", None)
            if declared is not None and (
                not policy.is_sha256(declared)
                or policy.canonical_json_sha256(core) != declared
            ):
                raise ValueError(f"{name} row {index} content hash changed")
            rows.append(value)
        return rows


    def _claim_identity(
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
            raise PermissionError("independent verifier claim identity changed")
        return int(opened.st_dev), int(opened.st_ino)


    def _load_reservation(
        claim_fd: int,
        parent_fd: int,
        directory_name: str,
        manifest_file_sha256: str,
        manifest_content_sha256: str,
    ) -> tuple[dict[str, Any], bytes, tuple[int, int]]:
        identity = _claim_identity(claim_fd, parent_fd, directory_name)
        raw = _read_relative(claim_fd, "reservation.json")
        value = _parse_json(raw, name="exact reservation in verifier")
        if (
            value.get("schema") != policy.EXACT_RESERVATION_SCHEMA
            or value.get("directory_identity") != list(identity)
            or value.get("execution_manifest_file_sha256") != manifest_file_sha256
            or value.get("execution_manifest_content_sha256")
            != manifest_content_sha256
            or value.get("retry_authorized") is not False
        ):
            raise PermissionError("independent verifier reservation changed")
        bindings = value.get("required_exact_bindings")
        if not isinstance(bindings, Mapping) or set(bindings) != set(
            policy.REQUIRED_BINDING_NAMES
        ) or any(not policy.is_sha256(item) for item in bindings.values()):
            raise PermissionError("independent verifier bindings are incomplete")
        return value, raw, identity


    def _load_preflight_first(reservation: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
        expected = reservation["required_exact_bindings"][
            "preflight_receipt_file_sha256"
        ]
        raw = _read_repo(policy.PREFLIGHT_RECEIPT_RELATIVE_PATH)
        _assert_hash(raw, expected, name="preflight receipt first verifier input")
        value = _parse_json(raw, name="preflight receipt in verifier")
        if (
            value.get("schema") != policy.PREFLIGHT_RECEIPT_SCHEMA
            or value.get("status") != "PASS"
            or value.get("payload_open_count") != 0
            or value.get("forbidden_open_count") != 0
            or value.get("device_contract") != policy.DEVICE_CONTRACT
        ):
            raise PermissionError("independent verifier preflight binding failed")
        return value, raw


    def _artifact_values(
        claim_fd: int,
    ) -> tuple[dict[str, bytes], dict[str, Any]]:
        raw: dict[str, bytes] = {}
        values: dict[str, Any] = {}
        json_paths = {
            "reservation.json",
            "source_review.json",
            "input_bindings.json",
            "preflight_receipt_binding.json",
            "schedule.json",
            "initialization.json",
            "arms/promoted_jepa/checkpoint_metrics.json",
            "arms/matched_no_jepa/matched_update_metrics.json",
            "selection.json",
            "calibration/promoted_jepa.json",
            "calibration/matched_no_jepa.json",
            "selection_role_ablation_diagnostic.json",
            "access_ledger.json",
            "training_record.json",
        }
        jsonl_paths = {
            "arms/promoted_jepa/training_trace.jsonl",
            "arms/matched_no_jepa/training_trace.jsonl",
        }
        for relative in policy.EXACT_INVENTORY[:-1]:
            payload = _read_relative(claim_fd, relative)
            raw[relative] = payload
            if relative in json_paths:
                values[relative] = _parse_json(payload, name=relative)
            elif relative in jsonl_paths:
                values[relative] = _parse_jsonl(payload, name=relative)
        return raw, values


    def _validate_static_artifacts(
        reservation: Mapping[str, Any],
        preflight: Mapping[str, Any],
        raw: Mapping[str, bytes],
        values: Mapping[str, Any],
    ) -> None:
        source_review = values["source_review.json"]
        input_bindings = values["input_bindings.json"]
        preflight_binding = values["preflight_receipt_binding.json"]
        bindings = reservation["required_exact_bindings"]
        expected_sources = {
            **policy.reviewed_source_bindings(),
            policy.RAW_SUPERVISION_BUILDER_RELATIVE_PATH: reservation[
                "required_exact_bindings"
            ]["development_raw_supervision_builder_source_sha256"],
            policy.RAW_SUPERVISION_AUDITOR_RELATIVE_PATH: reservation[
                "required_exact_bindings"
            ]["development_raw_supervision_auditor_source_sha256"],
        }
        implementation_names = {
            policy.POLICY_RELATIVE_PATH: "implementation_policy_source_sha256",
            policy.LOSS_ADAPTER_RELATIVE_PATH: "loss_adapter_source_sha256",
            policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH: "preflight_executor_source_sha256",
            policy.PREFLIGHT_VERIFIER_RELATIVE_PATH: "preflight_verifier_source_sha256",
            policy.EXACT_EXECUTOR_RELATIVE_PATH: "exact_executor_source_sha256",
            policy.EXACT_TRAINER_RELATIVE_PATH: "exact_trainer_source_sha256",
            policy.EXACT_VERIFIER_RELATIVE_PATH: "exact_verifier_source_sha256",
        }
        expected_sources.update(
            {
                relative: reservation["required_exact_bindings"][binding]
                for relative, binding in implementation_names.items()
            }
        )
        implementation_review_raw = _read_repo(
            policy.IMPLEMENTATION_REVIEW_RELATIVE_PATH
        )
        _assert_hash(
            implementation_review_raw,
            bindings["implementation_independent_review_file_sha256"],
            name="implementation independent review",
        )
        implementation_review = policy.validate_implementation_review(
            _parse_json(
                implementation_review_raw,
                name="implementation independent review",
            )
        )
        if implementation_review["reviewed_sources"] != {
            relative: bindings[binding]
            for relative, binding in implementation_names.items()
        }:
            raise PermissionError("implementation review source map changed")
        expected_source_review = policy.content_value(
            {
                "schema": policy.SOURCE_REVIEW_SCHEMA,
                "reviewed_sources": expected_sources,
                "camera_v13_dynamic_authority_bindings": {
                    policy.CAMERA_V13_SOURCE_REVIEW_RELATIVE_PATH: bindings[
                        "camera_v13_source_review_file_sha256"
                    ],
                    policy.CAMERA_V13_LADDER_PREREGISTRATION_RELATIVE_PATH: (
                        bindings[
                            "camera_v13_ladder_preregistration_file_sha256"
                        ]
                    ),
                    policy.CAMERA_V13_LADDER_REVIEW_RELATIVE_PATH: bindings[
                        "camera_v13_ladder_independent_review_file_sha256"
                    ],
                },
                "implementation_review_file_sha256": bindings[
                    "implementation_independent_review_file_sha256"
                ],
                "frozen_parent_closure": policy.reviewed_source_bindings(),
                "live_navigation_readiness_hash_authoritative": False,
            }
        )
        if source_review != expected_source_review:
            raise PermissionError("exact source-review closure changed")
        for relative, expected in expected_sources.items():
            source_raw = _read_repo(relative)
            _assert_hash(source_raw, expected, name=f"reviewed source {relative}")
        dynamic_camera_sources = {
            policy.CAMERA_V13_SOURCE_REVIEW_RELATIVE_PATH: bindings[
                "camera_v13_source_review_file_sha256"
            ],
            policy.CAMERA_V13_LADDER_PREREGISTRATION_RELATIVE_PATH: bindings[
                "camera_v13_ladder_preregistration_file_sha256"
            ],
            policy.CAMERA_V13_LADDER_REVIEW_RELATIVE_PATH: bindings[
                "camera_v13_ladder_independent_review_file_sha256"
            ],
        }
        dynamic_camera_raw: dict[str, bytes] = {}
        for relative, expected in dynamic_camera_sources.items():
            dynamic_camera_raw[relative] = _read_repo(relative)
            _assert_hash(
                dynamic_camera_raw[relative],
                expected,
                name=f"dynamic Camera authority {relative}",
            )
        ladder_review = _parse_json(
            dynamic_camera_raw[policy.CAMERA_V13_LADDER_REVIEW_RELATIVE_PATH],
            name="Camera ladder independent review",
        )
        if (
            ladder_review.get("verdict") != "PASS"
            or ladder_review.get("future_attempt_count") != 7
            or ladder_review.get("seed_20260710_n5_reexecution_authorized")
            is not False
            or ladder_review.get("preregistration_file_sha256")
            != bindings["camera_v13_ladder_preregistration_file_sha256"]
        ):
            raise PermissionError("dynamic Camera ladder review changed")
        preflight_raw = _read_repo(policy.PREFLIGHT_RECEIPT_RELATIVE_PATH)
        _assert_hash(
            preflight_raw,
            bindings["preflight_receipt_file_sha256"],
            name="preflight receipt recheck",
        )
        expected_preflight_binding = policy.content_value(
            {
                "schema": "lewm_go2_shared_jepa_v5_full_training_v3_preflight_binding_v1",
                "receipt": policy.artifact_binding(
                    policy.PREFLIGHT_RECEIPT_RELATIVE_PATH,
                    preflight_raw,
                    content_sha256=str(preflight["content_sha256"]),
                ),
                "first_post_reservation_input_open": True,
                "preflight_live_state_inherited": False,
                "preflight_rerun": False,
            }
        )
        if preflight_binding != expected_preflight_binding:
            raise PermissionError("exact preflight binding artifact changed")
        preflight_completed_raw = _read_repo(policy.PREFLIGHT_COMPLETED_RELATIVE_PATH)
        _assert_hash(
            preflight_completed_raw,
            bindings["preflight_completed_file_sha256"],
            name="preflight completion",
        )
        preflight_completed = _parse_json(
            preflight_completed_raw,
            name="preflight completion",
        )
        receipt_binding = preflight_completed.get(
            "artifacts_before_completion", {}
        ).get("gpu_smoke_receipt.json")
        if (
            preflight_completed.get("schema")
            != policy.PREFLIGHT_COMPLETION_SCHEMA
            or preflight_completed.get("status")
            != "completed_after_independent_reconstruction"
            or not isinstance(receipt_binding, Mapping)
            or receipt_binding.get("file_sha256")
            != bindings["preflight_receipt_file_sha256"]
        ):
            raise PermissionError("preflight completion binding changed")
        preflight_review_raw = _read_repo(
            policy.PREFLIGHT_INDEPENDENT_REVIEW_RELATIVE_PATH
        )
        _assert_hash(
            preflight_review_raw,
            bindings["preflight_independent_review_file_sha256"],
            name="preflight independent review",
        )
        _parse_json(preflight_review_raw, name="preflight independent review")
        raw_manifest_raw = _read_repo(policy.RAW_SUPERVISION_MANIFEST_RELATIVE_PATH)
        _assert_hash(
            raw_manifest_raw,
            bindings["development_raw_supervision_manifest_file_sha256"],
            name="raw manifest for input binding",
        )
        raw_manifest = _parse_json(
            raw_manifest_raw,
            name="raw manifest for input binding",
        )
        raw_audit_raw = _read_repo(policy.RAW_SUPERVISION_AUDIT_RELATIVE_PATH)
        _assert_hash(
            raw_audit_raw,
            bindings["development_raw_supervision_audit_file_sha256"],
            name="raw audit for input binding",
        )
        raw_audit = _parse_json(
            raw_audit_raw,
            name="raw audit for input binding",
        )
        camera_source_review_raw = _read_repo(
            policy.CAMERA_V13_SOURCE_REVIEW_RELATIVE_PATH
        )
        _assert_hash(
            camera_source_review_raw,
            bindings["camera_v13_source_review_file_sha256"],
            name="Camera V13 source review for input binding",
        )
        camera_source_review = _parse_json(
            camera_source_review_raw,
            name="Camera V13 source review for input binding",
        )
        camera_n5_gate_raw = _read_repo(policy.CAMERA_V13_N5_GATE_RELATIVE_PATH)
        _assert_hash(
            camera_n5_gate_raw,
            bindings["camera_v13_n5_gate_pass_file_sha256"],
            name="Camera V13 N5 gate for input binding",
        )
        camera_n5_gate = _parse_json(
            camera_n5_gate_raw,
            name="Camera V13 N5 gate for input binding",
        )
        v4_ladder_raw = _read_repo(policy.V4_TWO_SEED_LADDER_RELATIVE_PATH)
        _assert_hash(
            v4_ladder_raw,
            bindings["v4_two_seed_ladder_pass_file_sha256"],
            name="V4 ladder for input binding",
        )
        v4_ladder = _parse_json(
            v4_ladder_raw,
            name="V4 ladder for input binding",
        )
        if (
            raw_manifest["content_sha256"]
            != bindings["development_raw_supervision_manifest_content_sha256"]
            or raw_audit["content_sha256"]
            != bindings["development_raw_supervision_audit_content_sha256"]
            or raw_audit.get("schema") != policy.RAW_SUPERVISION_AUDIT_SCHEMA
            or raw_audit.get("verdict") != "PASS"
            or raw_audit.get("sample_results_sha256")
            != policy.RAW_V13_SAMPLE_RESULTS_SHA256
            or camera_source_review.get("content_sha256")
            != bindings["camera_v13_source_review_content_sha256"]
            or camera_source_review.get("source_closure_approved") is not True
            or camera_n5_gate.get("content_sha256")
            != bindings["camera_v13_n5_gate_pass_content_sha256"]
            or camera_n5_gate.get("passes") is not True
            or camera_n5_gate.get("seed") != 20260710
            or camera_n5_gate.get("fit_size") != 5
            or v4_ladder["content_sha256"]
            != bindings["v4_two_seed_ladder_pass_content_sha256"]
            or v4_ladder.get("rung_count") != 8
            or v4_ladder.get("additional_attempt_count") != 7
            or v4_ladder.get("seed_20260710_n5_reexecuted") is not False
            or v4_ladder.get("warm_start_used") is not False
        ):
            raise PermissionError("exact input semantic binding changed")
        expected_input_bindings = policy.content_value(
            {
                "schema": policy.INPUT_BINDINGS_SCHEMA,
                "execution_manifest_file_sha256": reservation[
                    "execution_manifest_file_sha256"
                ],
                "execution_manifest_content_sha256": reservation[
                    "execution_manifest_content_sha256"
                ],
                "preflight_receipt_file_sha256": bindings[
                    "preflight_receipt_file_sha256"
                ],
                "raw_manifest_file_sha256": bindings[
                    "development_raw_supervision_manifest_file_sha256"
                ],
                "raw_manifest_content_sha256": raw_manifest["content_sha256"],
                "raw_audit_file_sha256": bindings[
                    "development_raw_supervision_audit_file_sha256"
                ],
                "raw_audit_content_sha256": raw_audit["content_sha256"],
                "raw_v13_source_chain": policy.RAW_CHAIN_SOURCE_BINDINGS,
                "raw_v13_dataset_use_grant": policy.RAW_DATASET_USE_GRANT,
                "camera_v13_source_review_file_sha256": bindings[
                    "camera_v13_source_review_file_sha256"
                ],
                "camera_v13_source_review_content_sha256": (
                    camera_source_review["content_sha256"]
                ),
                "camera_v13_n5_gate_file_sha256": bindings[
                    "camera_v13_n5_gate_pass_file_sha256"
                ],
                "camera_v13_n5_gate_content_sha256": camera_n5_gate[
                    "content_sha256"
                ],
                "camera_v13_ladder_preregistration_file_sha256": bindings[
                    "camera_v13_ladder_preregistration_file_sha256"
                ],
                "camera_v13_ladder_independent_review_file_sha256": bindings[
                    "camera_v13_ladder_independent_review_file_sha256"
                ],
                "v4_two_seed_ladder_file_sha256": bindings[
                    "v4_two_seed_ladder_pass_file_sha256"
                ],
                "v4_two_seed_ladder_content_sha256": v4_ladder[
                    "content_sha256"
                ],
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
        )
        if input_bindings != expected_input_bindings:
            raise PermissionError("exact input-binding artifact changed")


    def _validate_traces(values: Mapping[str, Any]) -> None:
        loss_names = {
            "backward",
            "joint_total",
            "jepa_total",
            "jepa_prediction",
            "jepa_equivariance",
            "jepa_action_contrast",
            "jepa_variance",
            "jepa_warped_persistence",
            "pair_v4_total",
            "current_v4_hierarchical_first_hit_nll",
            "current_v4_target_bin_offset_smooth_l1",
            "current_v4_ground_clear_distance_state_balanced_bce",
            "current_v4_derived_raster_hierarchical_bce",
            "current_v4_derived_raster_cell_nll",
            "next_v4_hierarchical_first_hit_nll",
            "next_v4_target_bin_offset_smooth_l1",
            "next_v4_ground_clear_distance_state_balanced_bce",
            "next_v4_derived_raster_hierarchical_bce",
            "next_v4_derived_raster_cell_nll",
        }
        for arm in policy.ARMS:
            rows = values[f"arms/{arm}/training_trace.jsonl"]
            if len(rows) != policy.UPDATE_COUNT:
                raise PermissionError(f"{arm} trace update count changed")
            for index, row in enumerate(rows, start=1):
                if (
                    row.get("schema")
                    != "lewm_go2_shared_jepa_v5_full_training_v3_trace_row_v1"
                    or row.get("arm") != arm
                    or row.get("update") != index
                    or row.get("learning_rate") != policy.learning_rate(index)
                    or row.get("microbatch_count") != policy.ACCUMULATION_STEPS
                    or row.get("optimizer_step_count") != index
                    or row.get("ema_step_count") != index
                ):
                    raise PermissionError(f"{arm} trace cadence changed at {index}")
                losses = row.get("losses")
                if not isinstance(losses, Mapping) or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in losses.values()
                ) or set(losses) != loss_names:
                    raise ValueError(f"{arm} trace losses are nonfinite")
                current_v4 = 0.25 * sum(
                    float(losses[name])
                    for name in loss_names
                    if name.startswith("current_v4_")
                )
                next_v4 = 0.25 * sum(
                    float(losses[name])
                    for name in loss_names
                    if name.startswith("next_v4_")
                )
                expected_pair = 0.5 * (current_v4 + next_v4)
                if (
                    abs(float(losses["pair_v4_total"]) - expected_pair) > 1e-5
                    or abs(
                        float(losses["joint_total"])
                        - float(losses["jepa_total"])
                        - float(losses["pair_v4_total"])
                    )
                    > 1e-5
                    or not math.isfinite(float(row.get("gradient_norm_before_clip")))
                    or not math.isfinite(float(row.get("gradient_norm_after_clip")))
                    or float(row["gradient_norm_before_clip"]) < 0.0
                    or not 0.0 <= float(row["gradient_norm_after_clip"]) <= 1.000001
                ):
                    raise PermissionError(f"{arm} trace arithmetic changed at {index}")
                if arm == "promoted_jepa" and losses.get("backward") != losses.get(
                    "joint_total"
                ):
                    raise PermissionError("promoted backward loss changed")
                if arm == "matched_no_jepa" and losses.get("backward") != losses.get(
                    "pair_v4_total"
                ):
                    raise PermissionError("no-JEPA backward loss changed")


    def _load_fixed_backend() -> Any:
        if (
            os.environ.get("HIP_VISIBLE_DEVICES") != "0"
            or os.environ.get("ROCR_VISIBLE_DEVICES") != "0"
            or "HSA_OVERRIDE_GFX_VERSION" in os.environ
        ):
            raise PermissionError("independent verifier accelerator environment changed")
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
            shared_observable_camera_ray_jepa_v5_full_training_v3_loss
            as loss_adapter,
        )

        class IndependentReconstructor:
            def __init__(
                self,
                reservation: Mapping[str, Any],
                artifact_raw: Mapping[str, bytes],
                artifact_values: Mapping[str, Any],
            ) -> None:
                self.reservation = reservation
                self.artifact_raw = artifact_raw
                self.values = artifact_values
                self.cache: dict[str, bytes] = {}
                self.file_records: dict[str, Mapping[str, Any]] = {}
                self.device = self._device()

            def _device(self) -> Any:
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise PermissionError("independent reconstruction requires one ROCm GPU")
                properties = torch.cuda.get_device_properties(0)
                if (
                    str(properties.name) != policy.DEVICE_CONTRACT["device_name"]
                    or int(properties.total_memory)
                    < policy.DEVICE_CONTRACT["minimum_total_memory_bytes"]
                ):
                    raise PermissionError("independent reconstruction R9700 changed")
                torch.use_deterministic_algorithms(True)
                return torch.device("cuda:0")

            def _load_bound_json_inputs(self) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
                bindings = self.reservation["required_exact_bindings"]
                manifest_raw = _read_repo(policy.RAW_SUPERVISION_MANIFEST_RELATIVE_PATH)
                _assert_hash(
                    manifest_raw,
                    bindings["development_raw_supervision_manifest_file_sha256"],
                    name="independent raw manifest",
                )
                manifest = _parse_json(manifest_raw, name="independent raw manifest")
                audit_raw = _read_repo(policy.RAW_SUPERVISION_AUDIT_RELATIVE_PATH)
                _assert_hash(
                    audit_raw,
                    bindings["development_raw_supervision_audit_file_sha256"],
                    name="independent raw audit",
                )
                audit = _parse_json(audit_raw, name="independent raw audit")
                if (
                    manifest.get("content_sha256")
                    != bindings["development_raw_supervision_manifest_content_sha256"]
                    or audit.get("content_sha256")
                    != bindings["development_raw_supervision_audit_content_sha256"]
                    or audit.get("verdict") != "PASS"
                ):
                    raise PermissionError("independent raw authority changed")
                self.file_records = {
                    str(item["path"]): item
                    for item in manifest.get("files", [])
                    if isinstance(item, Mapping)
                }
                pairs = _parse_jsonl(self._raw_file("pairs.jsonl"), name="independent pairs")
                endpoints = _parse_jsonl(
                    self._raw_file("endpoints.jsonl"),
                    name="independent endpoints",
                )
                if (
                    manifest.get("schema")
                    != policy.RAW_SUPERVISION_MANIFEST_SCHEMA
                    or manifest.get("status")
                    != "complete_pending_independent_audit"
                    or manifest.get("roles") != list(policy.DEVELOPMENT_ROLES)
                    or manifest.get("pair_counts")
                    != {
                        role: policy.ROLE_COUNTS[role]["pairs"]
                        for role in policy.DEVELOPMENT_ROLES
                    }
                    or manifest.get("unique_endpoint_counts")
                    != {
                        role: policy.ROLE_COUNTS[role]["unique_endpoints"]
                        for role in policy.DEVELOPMENT_ROLES
                    }
                    or manifest.get("ordered_pair_sha256")
                    != policy.canonical_json_sha256(
                        [item["content_sha256"] for item in pairs]
                    )
                    or manifest.get("ordered_endpoint_sha256")
                    != policy.canonical_json_sha256(
                        [item["content_sha256"] for item in endpoints]
                    )
                    or audit.get("schema") != policy.RAW_SUPERVISION_AUDIT_SCHEMA
                    or audit.get("dataset_manifest_file_sha256")
                    != bindings[
                        "development_raw_supervision_manifest_file_sha256"
                    ]
                    or audit.get("dataset_manifest_content_sha256")
                    != manifest["content_sha256"]
                    or audit.get("g2_authorized") is not False
                    or audit.get("production_authorized") is not False
                ):
                    raise PermissionError("independent raw contract changed")
                endpoint_map = {
                    str(item["endpoint_identity_sha256"]): item
                    for item in endpoints
                }
                if (
                    len(endpoint_map) != len(endpoints)
                    or len({str(item["content_sha256"]) for item in pairs})
                    != len(pairs)
                ):
                    raise PermissionError("independent raw identities changed")
                for role in policy.DEVELOPMENT_ROLES:
                    role_pairs = [item for item in pairs if item["dataset_role"] == role]
                    role_endpoints = [
                        item for item in endpoints if item["dataset_role"] == role
                    ]
                    if (
                        len(role_pairs) != policy.ROLE_COUNTS[role]["pairs"]
                        or len(role_endpoints)
                        != policy.ROLE_COUNTS[role]["unique_endpoints"]
                        or len({str(item["scene_id"]) for item in role_pairs})
                        != policy.ROLE_COUNTS[role]["scenes"]
                        or {
                            str(item["family"])
                            for item in role_pairs
                        }
                        != set(policy.FAMILIES)
                    ):
                        raise PermissionError(f"independent raw role changed: {role}")
                if len(pairs) != sum(
                    policy.ROLE_COUNTS[role]["pairs"]
                    for role in policy.DEVELOPMENT_ROLES
                ) or len(endpoints) != sum(
                    policy.ROLE_COUNTS[role]["unique_endpoints"]
                    for role in policy.DEVELOPMENT_ROLES
                ):
                    raise PermissionError("independent raw role universe changed")
                for pair in pairs:
                    for side in ("current", "next"):
                        endpoint = endpoint_map.get(
                            str(pair[f"{side}_endpoint_sha256"])
                        )
                        if endpoint is None or any(
                            endpoint[name] != pair[name]
                            for name in ("dataset_role", "family", "scene_id")
                        ):
                            raise PermissionError(
                                "independent pair/endpoint boundary changed"
                            )
                return manifest, audit, pairs, endpoints

            def _raw_file(self, relative: str) -> bytes:
                cached = self.cache.get(relative)
                if cached is not None:
                    return cached
                record = self.file_records.get(relative)
                if not isinstance(record, Mapping):
                    raise PermissionError(f"independent raw file absent: {relative}")
                raw = _read_repo(f"{policy.RAW_SUPERVISION_ROOT_RELATIVE_PATH}/{relative}")
                _assert_hash(raw, str(record["file_sha256"]), name=f"raw {relative}")
                self.cache[relative] = raw
                return raw

            def _reconstruct_schedule(self, train_pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(policy.SCHEDULE_SEED)
                indices: list[int] = []
                while len(indices) < policy.PRESENTATION_COUNT:
                    indices.extend(torch.randperm(policy.TRAIN_PAIR_COUNT, generator=generator).tolist())
                indices = indices[: policy.PRESENTATION_COUNT]
                ordered_ids = [str(item["content_sha256"]) for item in train_pairs]
                commitment = policy.schedule_commitment(indices, ordered_ids)
                schedule = self.values["schedule.json"]
                presentations = [ordered_ids[index] for index in indices]
                if (
                    schedule.get("schema") != policy.SCHEDULE_SCHEMA
                    or schedule.get("ordered_train_pair_ids") != ordered_ids
                    or schedule.get("presentation_indices") != indices
                    or schedule.get("per_update_pair_ids")
                    != [
                        presentations[offset : offset + policy.EFFECTIVE_BATCH_SIZE]
                        for offset in range(0, policy.PRESENTATION_COUNT, policy.EFFECTIVE_BATCH_SIZE)
                    ]
                    or any(
                        schedule.get(name) != value
                        for name, value in commitment.items()
                        if name != "content_sha256"
                    )
                ):
                    raise PermissionError("independent schedule reconstruction differs")
                return schedule

            def _reconstruct_initialization(self) -> tuple[Mapping[str, Any], list[str], Any]:
                bindings = self.reservation["required_exact_bindings"]
                raw = _read_repo(policy.V4_PRIMARY_CHECKPOINT_RELATIVE_PATH)
                _assert_hash(
                    raw,
                    bindings["v4_primary_seed_20260710_n320_checkpoint_file_sha256"],
                    name="independent primary V4 checkpoint",
                )
                checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
                fit = ObservableCameraRayEvidenceV4Model()
                fit.load_state_dict(checkpoint["state_dict"], strict=True)
                torch.random.default_generator.manual_seed(
                    policy.INITIALIZATION_SEED
                )
                shared = model_module.SharedObservableCameraRayJepaV5()
                migration = shared.migrate_from_fit_model(fit)
                state_sha = model_module.tensor_state_dict_sha256(shared.state_dict())
                initialization = self.values["initialization.json"]
                expected_migration = {
                    "fit_model_state_sha256": migration.fit_model_state_sha256,
                    "shared_encoder_state_sha256": (
                        migration.shared_encoder_state_sha256
                    ),
                    "evidence_head_state_sha256": (
                        migration.evidence_head_state_sha256
                    ),
                    "migrated_head_key_count": migration.migrated_head_key_count,
                    "source_shape": list(migration.source_shape),
                    "pixel_ray_shape": list(migration.pixel_ray_shape),
                }
                if (
                    initialization.get("schema") != policy.INITIALIZATION_SCHEMA
                    or initialization.get("seed") != policy.INITIALIZATION_SEED
                    or initialization.get("primary_v4_seed")
                    != policy.PRIMARY_V4_SEED
                    or initialization.get("primary_v4_fit_size") != 320
                    or initialization.get("complete_training_state_sha256") != state_sha
                    or initialization.get("arm_initial_state_sha256")
                    != {arm: state_sha for arm in policy.ARMS}
                    or initialization.get("migration") != expected_migration
                    or initialization.get("hard_sync_count_before_training") != 1
                    or initialization.get("arms") != list(policy.ARMS)
                    or initialization.get("identical_before_optimizer_construction")
                    is not True
                    or initialization.get("device_of_initialization") != "cpu"
                    or initialization.get("precision") != "float32"
                    or not policy.is_sha256(
                        initialization.get("serialized_training_state_file_sha256")
                    )
                ):
                    raise PermissionError("independent V4 migration reconstruction differs")
                vocabulary = initialization.get("primitive_vocabulary")
                table = initialization.get("commanded_delta_table")
                if (
                    not isinstance(vocabulary, list)
                    or vocabulary != sorted(vocabulary)
                    or len(vocabulary) != 9
                    or policy.canonical_json_sha256(table)
                    != initialization.get("commanded_delta_table_sha256")
                ):
                    raise PermissionError("independent commanded table binding changed")
                return shared.state_dict(), vocabulary, torch.tensor(table, dtype=torch.float32, device=self.device)

            def _checkpoint(self, arm: str, update: int) -> tuple[Any, Mapping[str, Any]]:
                relative = f"arms/{arm}/checkpoints/update_{update}.pt"
                payload = torch.load(
                    io.BytesIO(self.artifact_raw[relative]),
                    map_location="cpu",
                    weights_only=False,
                )
                semantic = {
                    name: payload.get(name)
                    for name in (
                        "schema",
                        "arm",
                        "update",
                        "model_config",
                        "model_state_sha256",
                        "initialization_state_sha256",
                        "schedule_content_sha256",
                        "optimizer_contract",
                        "development_only",
                        "runtime_ready",
                    )
                }
                if (
                    set(payload)
                    != {
                        *semantic,
                        "content_sha256",
                        "model_state_dict",
                        "optimizer_state_dict",
                        "cpu_rng_state",
                        "gpu_rng_state",
                    }
                    or payload.get("schema")
                    != "lewm_go2_shared_jepa_v5_full_training_v3_checkpoint_v1"
                    or payload.get("arm") != arm
                    or payload.get("update") != update
                    or payload.get("model_config")
                    != model_module.SharedObservableCameraRayJepaV5Config().to_dict()
                    or payload.get("initialization_state_sha256")
                    != self.values["initialization.json"][
                        "complete_training_state_sha256"
                    ]
                    or payload.get("schedule_content_sha256")
                    != self.values["schedule.json"]["content_sha256"]
                    or payload.get("optimizer_contract") != policy.OPTIMIZER_CONTRACT
                    or payload.get("development_only") is not True
                    or payload.get("runtime_ready") is not False
                    or payload.get("content_sha256")
                    != policy.canonical_json_sha256(semantic)
                    or not isinstance(payload.get("optimizer_state_dict"), Mapping)
                ):
                    raise PermissionError(f"independent checkpoint contract changed: {relative}")
                state_sha = model_module.tensor_state_dict_sha256(payload["model_state_dict"])
                if state_sha != payload.get("model_state_sha256"):
                    raise PermissionError(f"checkpoint state hash changed: {relative}")
                model = model_module.SharedObservableCameraRayJepaV5().to(self.device).eval()
                model.load_state_dict(payload["model_state_dict"], strict=True)
                optimizer = payload["optimizer_state_dict"]
                groups = optimizer.get("param_groups")
                state = optimizer.get("state")
                trainable_count = sum(
                    parameter.requires_grad for parameter in model.parameters()
                )
                if (
                    not isinstance(groups, list)
                    or len(groups) != 1
                    or not isinstance(state, Mapping)
                    or len(groups[0].get("params", [])) != trainable_count
                    or groups[0].get("lr") != policy.learning_rate(update)
                    or tuple(groups[0].get("betas", ())) != (0.9, 0.999)
                    or groups[0].get("eps") != 1e-8
                    or groups[0].get("weight_decay") != 1e-4
                    or groups[0].get("amsgrad") is not False
                    or not set(state) <= set(groups[0]["params"])
                    or any(
                        float(value.get("step", -1)) != float(update)
                        for value in state.values()
                        if isinstance(value, Mapping)
                    )
                    or not isinstance(payload.get("cpu_rng_state"), torch.Tensor)
                    or not isinstance(payload.get("gpu_rng_state"), torch.Tensor)
                ):
                    raise PermissionError(
                        f"independent optimizer/RNG contract changed: {relative}"
                    )
                return model, payload

            def _endpoint(self, endpoint: Mapping[str, Any]) -> dict[str, Any]:
                shard_path = str(endpoint["scene_shard"])
                shard = _parse_json(self._raw_file(shard_path), name="independent shard")
                records = {str(item["path"]): item for item in shard["files"]}
                row = int(endpoint["shard_row"])
                def array(filename: str, dtype: str) -> Any:
                    relative = f"{Path(shard_path).parent}/{filename}"
                    record = records[filename]
                    raw = self._raw_file(relative)
                    _assert_hash(raw, str(record["file_sha256"]), name=relative)
                    values = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(tuple(record["shape"]))
                    return torch.from_numpy(values[row].copy()).to(self.device)
                image_relative = str(endpoint["image_path_metadata_only"])
                image_raw = _read_repo(image_relative)
                _assert_hash(
                    image_raw,
                    str(endpoint["image_sha256_commitment_only"]),
                    name="independent RGB",
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
                        raise ValueError("independent resized RGB shape changed")
                    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
                image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().to(self.device)
                mean = image.new_tensor(model_module.NORMALIZATION_MEAN)[:, None, None]
                std = image.new_tensor(model_module.NORMALIZATION_STD)[:, None, None]
                return {
                    "image": (image - mean) / std,
                    "origin": array("camera_origin_body_m.f4", "<f4"),
                    "basis": array("camera_basis_body_fru.f4", "<f4"),
                    "ground": array("ground_plane_z_body_m.f4", "<f4"),
                    "ground_in": array("ground_support_in_frustum.u1", "u1").bool(),
                    "ground_clear": array("ground_support_clear_to_target.u1", "u1").bool(),
                    "pixel_hit": array("pixel_hit_mask.u1", "u1").bool(),
                    "pixel_distance": array("pixel_first_hit_distance_m.f4", "<f4"),
                    "labels": array("raster_labels.u1", "u1").long(),
                }

            def _supervision(self, frames: Sequence[Mapping[str, Any]]) -> Any:
                stack = lambda name: torch.stack([item[name] for item in frames])
                return model_module.ObservableCameraRayV4FrameSupervisionV5(
                    pixel_hit_mask=stack("pixel_hit"),
                    pixel_first_hit_distance_m=stack("pixel_distance"),
                    ground_support_in_frustum=stack("ground_in"),
                    ground_support_clear_to_target=stack("ground_clear"),
                    target_raster_labels=stack("labels"),
                )

            def _pair_batch(self, pairs: Sequence[Mapping[str, Any]], endpoint_map: Mapping[str, Mapping[str, Any]], vocabulary: Sequence[str], table: Any) -> Mapping[str, Any]:
                current = [self._endpoint(endpoint_map[str(item["current_endpoint_sha256"])]) for item in pairs]
                next_frames = [self._endpoint(endpoint_map[str(item["next_endpoint_sha256"])]) for item in pairs]
                action_indices = [vocabulary.index(str(item["primitive"])) for item in pairs]
                action = torch.zeros((len(pairs), len(vocabulary)), device=self.device)
                action[torch.arange(len(pairs), device=self.device), torch.tensor(action_indices, device=self.device)] = 1.0
                wrong = torch.roll(action, 1, 1)
                realized = torch.tensor([item["relative_se2_current_frame"] for item in pairs], dtype=torch.float32, device=self.device)
                stack = lambda frames, name: torch.stack([item[name] for item in frames])
                return {
                    "forward": {
                        "current_image": stack(current, "image"),
                        "next_image": stack(next_frames, "image"),
                        "action": action,
                        "realized_delta_pose_current": realized,
                        "commanded_delta_pose_current": action @ table,
                        "current_camera_origin_body_m": stack(current, "origin"),
                        "current_camera_basis_body_fru": stack(current, "basis"),
                        "current_ground_plane_z_body_m": stack(current, "ground"),
                        "next_camera_origin_body_m": stack(next_frames, "origin"),
                        "next_camera_basis_body_fru": stack(next_frames, "basis"),
                        "next_ground_plane_z_body_m": stack(next_frames, "ground"),
                        "next_prediction_mask": torch.ones((len(pairs), 64, 64), dtype=torch.bool, device=self.device),
                        "diagnostic_wrong_action": wrong,
                        "diagnostic_wrong_action_delta_pose_current": wrong @ table,
                        "diagnostic_wrong_commanded_delta_pose_current": -(action @ table),
                    },
                    "current": current,
                    "next": next_frames,
                }

            def _jepa_scope(self, model: Any, pairs: Sequence[Mapping[str, Any]], endpoint_map: Mapping[str, Mapping[str, Any]], vocabulary: Sequence[str], table: Any) -> dict[str, Any]:
                numerators: dict[str, float] = {}
                denominators: dict[str, int] = {}
                target_parts: list[Any] = []

                def normalized_error(prediction: Any, target: Any) -> Any:
                    return (
                        F.normalize(prediction, dim=1)
                        - F.normalize(target, dim=1)
                    ).square().mean(dim=1)

                def add(name: str, values: Any, mask: Any) -> None:
                    if values.shape != mask.shape:
                        raise ValueError(
                            "independent raw JEPA accumulator shape changed"
                        )
                    weight = mask.to(values.dtype)
                    numerators[name] = numerators.get(name, 0.0) + float(
                        (values * weight).sum().cpu()
                    )
                    denominators[name] = denominators.get(name, 0) + int(
                        mask.sum().cpu()
                    )

                def average(name: str) -> float:
                    return numerators.get(name, 0.0) / max(
                        1,
                        denominators.get(name, 0),
                    )

                with torch.no_grad():
                    for start in range(0, len(pairs), policy.MICROBATCH_SIZE):
                        selected = pairs[start : start + policy.MICROBATCH_SIZE]
                        batch = self._pair_batch(
                            selected,
                            endpoint_map,
                            vocabulary,
                            table,
                        )
                        pair = model.forward_training_pair(**batch["forward"])
                        package = pair.jepa
                        if any(
                            not bool(torch.isfinite(value).item())
                            for value in (
                                package.total,
                                package.prediction,
                                package.equivariance,
                                package.action_contrast,
                                package.variance,
                                package.warped_persistence,
                            )
                        ):
                            raise FloatingPointError(
                                "independent JEPA package is nonfinite"
                            )
                        target = pair.stop_gradient_target_next_bev.detach()
                        prediction = pair.predicted_next_bev.detach()
                        persistence = pair.commanded_warped_current_bev.detach()
                        prediction_mask = pair.commanded_overlap_mask[:, 0].bool()
                        prediction_error = normalized_error(prediction, target)
                        persistence_error = normalized_error(persistence, target)
                        add("prediction", prediction_error, prediction_mask)
                        add("persistence", persistence_error, prediction_mask)

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
                        add("wrong_action_real", prediction_error, wrong_action_mask)
                        add(
                            "wrong_action_persistence",
                            persistence_error,
                            wrong_action_mask,
                        )
                        add(
                            "wrong_action",
                            normalized_error(wrong_action_prediction, target),
                            wrong_action_mask,
                        )
                        add(
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
                        add("wrong_delta_real", prediction_error, wrong_delta_mask)
                        add(
                            "wrong_delta_persistence",
                            persistence_error,
                            wrong_delta_mask,
                        )
                        add(
                            "wrong_delta",
                            normalized_error(wrong_delta_prediction, target),
                            wrong_delta_mask,
                        )
                        add(
                            "wrong_delta_sensitivity",
                            normalized_error(wrong_delta_prediction, prediction),
                            wrong_delta_mask,
                        )
                        target_parts.append(target.cpu())

                if not target_parts:
                    raise ValueError("independent JEPA target population is empty")
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
                        samples = samples[
                            :: math.ceil(samples.shape[0] / 65_536)
                        ]
                    covariance = samples.T @ samples / max(
                        1,
                        samples.shape[0] - 1,
                    )
                    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
                    total = eigenvalues.sum()
                    if bool((total > 0).item()):
                        probabilities = eigenvalues / total
                        entropy = -(
                            probabilities
                            * probabilities.clamp_min(1e-12).log()
                        ).sum()
                        target_rank = float(torch.exp(entropy))
                    else:
                        target_rank = 0.0

                prediction_mean = average("prediction")
                persistence_mean = average("persistence")
                wrong_action_advantage = average("wrong_action") - average(
                    "wrong_action_real"
                )
                wrong_delta_advantage = average("wrong_delta") - average(
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
                        / max(average("wrong_action_persistence"), 1e-8)
                    ),
                    "wrong_commanded_delta_advantage_over_target_change": (
                        wrong_delta_advantage
                        / max(average("wrong_delta_persistence"), 1e-8)
                    ),
                    "wrong_action_prediction_sensitivity": average(
                        "wrong_action_sensitivity"
                    ),
                    "wrong_commanded_delta_prediction_sensitivity": average(
                        "wrong_delta_sensitivity"
                    ),
                }

            def _physical_scope(self, model: Any, pairs: Sequence[Mapping[str, Any]], endpoint_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
                ids = sorted({str(item[f"{side}_endpoint_sha256"]) for item in pairs for side in ("current", "next")})
                frames = [self._endpoint(endpoint_map[digest]) for digest in ids]
                families = [str(endpoint_map[digest]["family"]) for digest in ids]
                groups: dict[str, list[int]] = {}
                for index, family in enumerate(families):
                    groups.setdefault(family, []).append(index)
                wrong_map = list(range(len(frames)))
                for members in groups.values():
                    for offset, index in enumerate(members):
                        wrong_map[index] = members[(offset + 1) % len(members)]
                correct = ObservableCameraRayFitV4MetricAccumulator()
                wrong = ObservableCameraRayFitV4MetricAccumulator()
                complete_loss = 0.0
                with torch.no_grad():
                    for start in range(0, len(frames), policy.MICROBATCH_SIZE):
                        indices = list(range(start, min(start + policy.MICROBATCH_SIZE, len(frames))))
                        target = [frames[index] for index in indices]
                        supervision = self._supervision(target)
                        for source, accumulator in ((target, correct), ([frames[wrong_map[index]] for index in indices], wrong)):
                            online = model.forward_frame(
                                torch.stack([item["image"] for item in source]),
                                torch.stack([item["origin"] for item in target]),
                                torch.stack([item["basis"] for item in target]),
                                torch.stack([item["ground"] for item in target]),
                            )
                            targets = derive_observable_camera_ray_evidence_v4_targets(
                                pixel_hit_mask=supervision.pixel_hit_mask,
                                pixel_first_hit_distance_m=supervision.pixel_first_hit_distance_m,
                                ground_support_in_frustum=supervision.ground_support_in_frustum,
                                ground_support_clear_to_target=supervision.ground_support_clear_to_target,
                            )
                            soft = soft_rasterize_observable_camera_ray_evidence_v4(
                                online.evidence,
                                camera_origin_body_m=online.camera_origin_body_m,
                                camera_basis_body_fru=online.camera_basis_body_fru,
                            )
                            accumulator.update(
                                raw_output=online.evidence,
                                targets=targets,
                                soft_raster=soft,
                                target_raster_labels=supervision.target_raster_labels,
                                families=[families[index] for index in indices],
                            )
                            if accumulator is correct:
                                fake_pair = model_module.SharedTrainingPairV5(
                                    current=online,
                                    next=online,
                                    predicted_next_bev=online.bev,
                                    stop_gradient_target_next_bev=online.bev.detach(),
                                    commanded_warped_current_bev=online.bev,
                                    commanded_overlap_mask=torch.ones_like(online.bev[:, :1], dtype=torch.bool),
                                    realized_warped_current_bev=online.bev,
                                    realized_overlap_mask=torch.ones_like(online.bev[:, :1], dtype=torch.bool),
                                    jepa=None,
                                )
                                complete_loss += float(
                                    loss_adapter.observable_camera_ray_v4_loss_v3(
                                        model,
                                        fake_pair,
                                        supervision,
                                        supervision,
                                        require_b4=False,
                                    ).total.cpu()
                                ) * len(indices)
                return self._flatten_physical(correct.finalize(), wrong.finalize(), complete_loss / len(frames))

            @staticmethod
            def _flatten_physical(correct: Mapping[str, Any], wrong: Mapping[str, Any], loss: float) -> dict[str, Any]:
                c_depth, w_depth = correct["pixel_hit_depth"], wrong["pixel_hit_depth"]
                c_raster, w_raster = correct["derived_raster"], wrong["derived_raster"]
                return {
                    "pixel_first_hit_balanced_accuracy": correct["pixel_hit_no_hit"]["balanced_accuracy"],
                    "depth_median_error_m": c_depth["median_absolute_error_m"],
                    "depth_p95_error_m": c_depth["p95_absolute_error_m"],
                    "ground_clear_balanced_accuracy": correct["ground_clear"]["overall"]["balanced_accuracy"],
                    "distance_group_balanced_accuracy": [value["balanced_accuracy"] for value in correct["ground_clear"]["by_distance_m"].values() if value["count"] > 0],
                    "derived_raster_nll": c_raster["nll"],
                    "derived_raster_balanced_accuracy": c_raster["balanced_accuracy"],
                    "present_class_recall": {name: value for name, value in c_raster["class_recalls"].items() if value is not None},
                    "wrong_rgb_pixel_balanced_accuracy_drop": correct["pixel_hit_no_hit"]["balanced_accuracy"] - wrong["pixel_hit_no_hit"]["balanced_accuracy"],
                    "wrong_rgb_depth_median_error_increase_m": w_depth["median_absolute_error_m"] - c_depth["median_absolute_error_m"],
                    "wrong_rgb_depth_p95_error_increase_m": w_depth["p95_absolute_error_m"] - c_depth["p95_absolute_error_m"],
                    "wrong_rgb_ground_balanced_accuracy_drop": correct["ground_clear"]["overall"]["balanced_accuracy"] - wrong["ground_clear"]["overall"]["balanced_accuracy"],
                    "wrong_rgb_raster_nll_increase": w_raster["nll"] - c_raster["nll"],
                    "wrong_rgb_raster_balanced_accuracy_drop": c_raster["balanced_accuracy"] - w_raster["balanced_accuracy"],
                    "complete_v4_loss": loss,
                }

            def _candidate(self, model: Any, update: int, pairs: Sequence[Mapping[str, Any]], endpoints: Mapping[str, Mapping[str, Any]], vocabulary: Sequence[str], table: Any) -> dict[str, Any]:
                scopes = {}
                for scope in policy.SCOPES:
                    selected = list(pairs) if scope == "aggregate" else [item for item in pairs if item["family"] == scope]
                    scopes[scope] = {
                        "physical": self._physical_scope(model, selected, endpoints),
                        "jepa": self._jepa_scope(model, selected, endpoints, vocabulary, table),
                    }
                return {
                    "update": update,
                    "scopes": scopes,
                    "aggregate_complete_v4_loss": scopes["aggregate"]["physical"]["complete_v4_loss"],
                    "aggregate_prediction_to_persistence_ratio": scopes["aggregate"]["jepa"]["prediction_to_warped_persistence_ratio"],
                }

            def _calibration(self, model: Any, pairs: Sequence[Mapping[str, Any]], endpoint_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
                ids = sorted({str(item[f"{side}_endpoint_sha256"]) for item in pairs for side in ("current", "next")})
                logits_parts, label_parts, within_parts = [], [], []
                family_indices: dict[str, list[int]] = {family: [] for family in policy.FAMILIES}
                config = model.model_config
                rows, columns = config.bev_size
                x = config.forward_range_m[0] + (torch.arange(rows) + 0.5) * ((config.forward_range_m[1] - config.forward_range_m[0]) / rows)
                y = config.left_range_m[0] + (torch.arange(columns) + 0.5) * ((config.left_range_m[1] - config.left_range_m[0]) / columns)
                gx, gy = torch.meshgrid(x, y, indexing="ij")
                frame_within = (gx.square() + gy.square()).sqrt().reshape(-1) <= 2.0
                with torch.no_grad():
                    for digest in ids:
                        frame = self._endpoint(endpoint_map[digest])
                        online = model.forward_frame(frame["image"][None], frame["origin"][None], frame["basis"][None], frame["ground"][None])
                        soft = soft_rasterize_observable_camera_ray_evidence_v4(online.evidence, camera_origin_body_m=online.camera_origin_body_m, camera_basis_body_fru=online.camera_basis_body_fru)
                        logits = soft.class_probabilities.clamp_min(torch.finfo(torch.float32).eps).log().permute(0, 2, 3, 1).reshape(-1, 3).cpu()
                        labels = frame["labels"].reshape(-1).cpu()
                        base = sum(item.numel() for item in label_parts)
                        family_indices[str(endpoint_map[digest]["family"])].extend(range(base, base + labels.numel()))
                        logits_parts.append(logits)
                        label_parts.append(labels)
                        within_parts.append(frame_within)
                logits, labels, within = torch.cat(logits_parts), torch.cat(label_parts), torch.cat(within_parts)
                class_counts = torch.bincount(labels, minlength=3)
                if bool((class_counts == 0).any().item()):
                    raise ValueError("independent calibration role is missing a class")
                raw = torch.zeros(6, dtype=torch.float32, requires_grad=True)
                optimizer = torch.optim.LBFGS((raw,), lr=0.5, max_iter=80, line_search_fn="strong_wolfe")
                before = float(F.cross_entropy(logits, labels).item())
                if not math.isfinite(before):
                    raise FloatingPointError(
                        "independent uncalibrated NLL is nonfinite"
                    )
                def scaled(parameters: Any) -> Any:
                    return logits * parameters[:3].clamp(-3, 3).exp()[None] + (parameters[3:] - parameters[3:].mean())[None]
                def closure() -> Any:
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(scaled(raw), labels)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                params = policy.centered_vector_scaling_parameters(raw[:3].detach().tolist(), raw[3:].detach().tolist())
                fixed = torch.tensor(params["log_scales"] + params["centered_biases"])
                probabilities = scaled(fixed).softmax(1)
                after = float(F.cross_entropy(scaled(fixed), labels).item())
                if not math.isfinite(after):
                    raise FloatingPointError(
                        "independent calibrated NLL is nonfinite"
                    )
                aggregate_mask = torch.ones(labels.numel(), dtype=torch.bool)
                reports = {}
                for values in policy.threshold_grid():
                    reports[policy.canonical_json_sha256(list(values))] = self._threshold_counts(probabilities, labels, within, values)
                threshold = policy.select_calibration_threshold(reports)
                scope_masks = {"aggregate": aggregate_mask}
                for family, indices in family_indices.items():
                    mask = torch.zeros(labels.numel(), dtype=torch.bool)
                    mask[indices] = True
                    scope_masks[family] = mask
                scope_reports = {}
                calibrated_logits = scaled(fixed)
                for scope, mask in scope_masks.items():
                    report = self._fixed_report(
                        probabilities[mask],
                        labels[mask],
                        within[mask],
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
                            "independent calibration scope NLL is nonfinite: "
                            + scope
                        )
                    scope_reports[scope] = report
                return {"parameters": params, "uncalibrated_nll": before, "calibrated_nll": after, "class_counts": class_counts.tolist(), "threshold": threshold, "scope_reports": scope_reports}

            @staticmethod
            def _threshold_counts(probabilities: Any, labels: Any, within: Any, values: Sequence[float]) -> dict[str, int]:
                free_min, occupied_max, unknown_max, detection_min = values
                admitted = (probabilities[:, 1] >= free_min) & (probabilities[:, 2] <= occupied_max) & (probabilities[:, 0] <= unknown_max)
                free = labels == 1
                obstacle = (labels == 2) & within
                detected = probabilities[:, 2] >= detection_min
                return {"admitted_free_count": int(admitted.sum()), "admitted_free_true_free_count": int((admitted & free).sum()), "useful_free_count": int(free.sum()), "useful_free_admitted_count": int((free & admitted).sum()), "obstacle_within_2m_count": int(obstacle.sum()), "obstacle_within_2m_excluded_count": int((obstacle & ~admitted).sum()), "obstacle_within_2m_detected_count": int((obstacle & detected).sum())}

            def _fixed_report(self, probabilities: Any, labels: Any, within: Any, threshold: Mapping[str, Any]) -> dict[str, Any]:
                counts = self._threshold_counts(probabilities, labels, within, (threshold["free_probability_minimum"], threshold["occupied_probability_maximum"], threshold["unknown_probability_maximum"], threshold["occupied_detection_minimum"]))
                admitted, useful, obstacles = counts["admitted_free_count"], counts["useful_free_count"], counts["obstacle_within_2m_count"]
                return {**counts, "admitted_free_precision": counts["admitted_free_true_free_count"] / admitted if admitted else None, "useful_free_recall": counts["useful_free_admitted_count"] / useful if useful else None, "obstacle_exclusion_recall_within_2m": counts["obstacle_within_2m_excluded_count"] / obstacles if obstacles else None, "obstacle_detection_recall_within_2m": counts["obstacle_within_2m_detected_count"] / obstacles if obstacles else None}

            def _metric_delta(self, promoted: Any, matched: Any) -> Any:
                if isinstance(promoted, Mapping) and isinstance(matched, Mapping):
                    if set(promoted) != set(matched):
                        raise ValueError("independent ablation metric fields changed")
                    return {
                        name: self._metric_delta(promoted[name], matched[name])
                        for name in promoted
                    }
                if (
                    isinstance(promoted, Sequence)
                    and not isinstance(promoted, (str, bytes))
                    and isinstance(matched, Sequence)
                    and not isinstance(matched, (str, bytes))
                ):
                    if len(promoted) != len(matched):
                        raise ValueError(
                            "independent ablation metric sequence changed"
                        )
                    return [
                        self._metric_delta(left, right)
                        for left, right in zip(promoted, matched, strict=True)
                    ]
                if (
                    isinstance(promoted, bool)
                    or isinstance(matched, bool)
                    or not isinstance(promoted, (int, float))
                    or not isinstance(matched, (int, float))
                ):
                    raise TypeError(
                        "independent ablation metric leaf is not numeric"
                    )
                delta = float(promoted) - float(matched)
                if not math.isfinite(delta):
                    raise FloatingPointError(
                        "independent ablation metric delta is nonfinite"
                    )
                return delta

            def _independent_artifact_bindings(self) -> dict[str, Any]:
                bindings: dict[str, Any] = {}
                for relative in policy.EXACT_INVENTORY[:-1]:
                    raw = self.artifact_raw[relative]
                    content_sha256 = None
                    state_sha256 = None
                    if relative in self.values and isinstance(
                        self.values[relative], Mapping
                    ):
                        content_sha256 = self.values[relative].get("content_sha256")
                    elif relative.endswith("training_trace.jsonl"):
                        content_sha256 = policy.canonical_json_sha256(
                            self.values[relative]
                        )
                    elif "/checkpoints/update_" in relative:
                        payload = torch.load(
                            io.BytesIO(raw),
                            map_location="cpu",
                            weights_only=False,
                        )
                        content_sha256 = payload.get("content_sha256")
                        state_sha256 = model_module.tensor_state_dict_sha256(
                            payload["model_state_dict"]
                        )
                    elif relative == "pre_g2_candidate_checkpoint.pt":
                        payload = torch.load(
                            io.BytesIO(raw),
                            map_location="cpu",
                            weights_only=False,
                        )
                        content_sha256 = payload.get("content_sha256")
                        state_sha256 = model_module.tensor_state_dict_sha256(
                            payload["deployment_state_dict"]
                        )
                    binding = policy.artifact_binding(
                        relative,
                        raw,
                        content_sha256=content_sha256,
                    )
                    if state_sha256 is not None:
                        binding["state_sha256"] = state_sha256
                    bindings[relative] = binding
                return bindings

            def reconstruct(self) -> dict[str, Any]:
                _manifest, _audit, pairs, endpoint_rows = self._load_bound_json_inputs()
                endpoint_map = {str(item["endpoint_identity_sha256"]): item for item in endpoint_rows}
                train = [item for item in pairs if item["dataset_role"] == "train"]
                selection_pairs = [item for item in pairs if item["dataset_role"] == "checkpoint_selection"]
                calibration_pairs = [item for item in pairs if item["dataset_role"] == "probability_calibration"]
                schedule = self._reconstruct_schedule(train)
                initial_state, vocabulary, table = self._reconstruct_initialization()
                expected_table = []
                for primitive in vocabulary:
                    rows = torch.tensor([item["relative_se2_current_frame"] for item in train if item["primitive"] == primitive], dtype=torch.float32)
                    expected_table.append(torch.quantile(rows, 0.5, dim=0))
                if torch.stack(expected_table).tolist() != self.values["initialization.json"]["commanded_delta_table"]:
                    raise PermissionError("independent commanded median table differs")
                migration_model = model_module.SharedObservableCameraRayJepaV5().to(
                    self.device
                ).eval()
                migration_model.load_state_dict(initial_state, strict=True)
                migration_baseline = self._candidate(
                    migration_model,
                    0,
                    selection_pairs,
                    endpoint_map,
                    vocabulary,
                    table,
                )
                del migration_model
                candidates = []
                for update in policy.CHECKPOINT_UPDATES:
                    model, checkpoint = self._checkpoint("promoted_jepa", update)
                    if checkpoint.get("schedule_content_sha256") != schedule["content_sha256"]:
                        raise PermissionError("checkpoint schedule binding changed")
                    candidates.append(self._candidate(model, update, selection_pairs, endpoint_map, vocabulary, table))
                checkpoint_metrics = self.values[
                    "arms/promoted_jepa/checkpoint_metrics.json"
                ]
                expected_checkpoint_metrics = policy.content_value(
                    {
                        "schema": "lewm_go2_shared_jepa_v5_full_training_v3_checkpoint_metrics_v1",
                        "role": "checkpoint_selection",
                        "pair_count": policy.ROLE_COUNTS[
                            "checkpoint_selection"
                        ]["pairs"],
                        "unique_endpoint_count": policy.ROLE_COUNTS[
                            "checkpoint_selection"
                        ]["unique_endpoints"],
                        "migration_baseline_nonselectable": migration_baseline,
                        "candidates": candidates,
                    }
                )
                if checkpoint_metrics != expected_checkpoint_metrics:
                    raise PermissionError("independently recomputed selection metrics differ")
                selected = policy.select_promoted_checkpoint(candidates)
                selection_value = self.values["selection.json"]
                expected_selection = policy.content_value(
                    {
                        "schema": policy.SELECTION_SCHEMA,
                        **selected,
                        "checkpoint_metrics_content_sha256": checkpoint_metrics[
                            "content_sha256"
                        ],
                        "ablation_influenced_selection": False,
                        "calibration_influenced_selection": False,
                    }
                )
                if selection_value != expected_selection:
                    raise PermissionError("independent promoted selection differs")
                selected_update = int(selected["selected_update"])
                matched_model = None
                for update in policy.CHECKPOINT_UPDATES:
                    candidate_model, _checkpoint = self._checkpoint(
                        "matched_no_jepa", update
                    )
                    if update == selected_update:
                        matched_model = candidate_model
                    else:
                        del candidate_model
                if matched_model is None:
                    raise RuntimeError("selected matched checkpoint was not reopened")
                matched_metrics = self._candidate(matched_model, selected_update, selection_pairs, endpoint_map, vocabulary, table)
                matched_value = self.values[
                    "arms/matched_no_jepa/matched_update_metrics.json"
                ]
                expected_matched = policy.content_value(
                    {
                        "schema": "lewm_go2_shared_jepa_v5_full_training_v3_matched_metrics_v1",
                        "selected_promoted_update": selected_update,
                        "metrics": matched_metrics,
                        "selection_effect": "none",
                    }
                )
                if matched_value != expected_matched:
                    raise PermissionError("independent matched diagnostic metrics differ")
                diagnostic_value = self.values[
                    "selection_role_ablation_diagnostic.json"
                ]
                scene_id_by_family = {}
                for family in policy.FAMILIES:
                    scene_ids = {
                        str(item["scene_id"])
                        for item in selection_pairs
                        if item["family"] == family
                    }
                    if len(scene_ids) != 1:
                        raise PermissionError(
                            "independent selection family is not one scene: "
                            + family
                        )
                    scene_id_by_family[family] = next(iter(scene_ids))
                promoted_selected_metrics = candidates[
                    policy.CHECKPOINT_UPDATES.index(selected_update)
                ]
                expected_diagnostic = policy.content_value(
                    {
                        "schema": policy.DIAGNOSTIC_ABLATION_SCHEMA,
                        **policy.selection_role_ablation_contract(),
                        "scene_id_by_family": scene_id_by_family,
                        "promoted": promoted_selected_metrics,
                        "matched_no_jepa": matched_metrics,
                        "raw_delta_direction": "promoted_minus_matched_no_jepa",
                        "raw_metric_deltas": self._metric_delta(
                            promoted_selected_metrics["scopes"],
                            matched_metrics["scopes"],
                        ),
                    }
                )
                if diagnostic_value != expected_diagnostic:
                    raise PermissionError("independent diagnostic artifact differs")
                promoted_model, _promoted = self._checkpoint("promoted_jepa", selected_update)
                calibrations = {
                    "promoted_jepa": self._calibration(promoted_model, calibration_pairs, endpoint_map),
                    "matched_no_jepa": self._calibration(matched_model, calibration_pairs, endpoint_map),
                }
                for arm, reconstructed in calibrations.items():
                    published = self.values[f"calibration/{arm}.json"]
                    expected_calibration = policy.content_value(
                        {
                            "schema": policy.CALIBRATION_SCHEMA,
                            "arm": arm,
                            "role": "probability_calibration",
                            "pair_count": policy.ROLE_COUNTS[
                                "probability_calibration"
                            ]["pairs"],
                            "unique_endpoint_count": policy.ROLE_COUNTS[
                                "probability_calibration"
                            ]["unique_endpoints"],
                            **reconstructed,
                        }
                    )
                    if published != expected_calibration:
                        raise PermissionError(f"independent {arm} calibration differs")
                promoted_calibration = calibrations["promoted_jepa"]
                if (
                    promoted_calibration["calibrated_nll"]
                    > promoted_calibration["uncalibrated_nll"] + 1e-6
                ):
                    raise PermissionError(
                        "independent promoted aggregate calibration worsened NLL"
                    )
                for scope, report in promoted_calibration[
                    "scope_reports"
                ].items():
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
                        raise PermissionError(
                            "independent promoted calibration gate failed: "
                            + scope
                        )
                candidate = torch.load(
                    io.BytesIO(
                        self.artifact_raw["pre_g2_candidate_checkpoint.pt"]
                    ),
                    map_location="cpu",
                    weights_only=False,
                )
                deployment = promoted_model.deployment_state_dict()
                deployment_sha = model_module.tensor_state_dict_sha256(deployment)
                candidate_state_sha = model_module.tensor_state_dict_sha256(
                    candidate["deployment_state_dict"]
                )
                candidate_core = {
                    name: value
                    for name, value in candidate.items()
                    if name not in {"content_sha256", "deployment_state_dict"}
                }
                expected_candidate_core = policy.pre_g2_candidate_checkpoint_core(
                    model_config=promoted_model.model_config.to_dict(),
                    deployment_state_sha256=deployment_sha,
                    selection=selection_value,
                    calibration=self.values["calibration/promoted_jepa.json"],
                )
                if (
                    set(candidate)
                    != {
                        *expected_candidate_core,
                        "content_sha256",
                        "deployment_state_dict",
                    }
                    or candidate_core != expected_candidate_core
                    or candidate.get("content_sha256")
                    != policy.canonical_json_sha256(candidate_core)
                    or candidate_state_sha != deployment_sha
                ):
                    raise PermissionError(
                        "independent pre-G2 candidate filtering differs"
                    )
                ledger = self.values["access_ledger.json"]
                summary = policy.validate_access_ledger(
                    ledger["events"],
                    require_completion_rehash=True,
                )
                if (
                    ledger.get("forbidden_open_count") != 0
                    or ledger.get("g2_open_count") != 0
                    or ledger.get("heldout_open_count") != 0
                    or ledger.get("runtime_navigation_hardware_open_count") != 0
                    or ledger.get("production_or_promotion_open_count") != 0
                    or ledger.get("summary") != summary
                ):
                    raise PermissionError("independent ledger reconstruction failed")
                device_properties = torch.cuda.get_device_properties(self.device)
                training_record = self.values["training_record.json"]
                expected_training_record = policy.content_value(
                    {
                        "schema": policy.TRAINING_RECORD_SCHEMA,
                        "status": "pre_g2_development_candidate_pending_independent_verification",
                        "device": {
                            "device": "cuda:0",
                            "name": str(device_properties.name),
                            "total_memory_bytes": int(device_properties.total_memory),
                            "visible_device_count": 1,
                            "torch_version": str(torch.__version__),
                            "hip_version": str(torch.version.hip),
                        },
                        "initialization_content_sha256": self.values[
                            "initialization.json"
                        ]["content_sha256"],
                        "schedule_content_sha256": schedule["content_sha256"],
                        "selection_content_sha256": selection_value[
                            "content_sha256"
                        ],
                        "calibration_content_sha256": {
                            arm: self.values[f"calibration/{arm}.json"][
                                "content_sha256"
                            ]
                            for arm in policy.ARMS
                        },
                        "diagnostic_content_sha256": diagnostic_value[
                            "content_sha256"
                        ],
                        "access_ledger_content_sha256": ledger["content_sha256"],
                        "optimizer_contract": policy.OPTIMIZER_CONTRACT,
                        "joint_loss_contract": policy.JOINT_LOSS_CONTRACT,
                        "runtime_ready": False,
                        "g2_authorized": False,
                        "heldout_authorized": False,
                        "production_or_promotion_authorized": False,
                        "retry_authorized": False,
                    }
                )
                if training_record != expected_training_record:
                    raise PermissionError("independent training record differs")
                artifact_bindings = self._independent_artifact_bindings()
                return {
                    "selected_update": selected_update,
                    "pre_g2_candidate_state_sha256": deployment_sha,
                    "schedule_content_sha256": schedule["content_sha256"],
                    "ledger_terminal_event_sha256": summary["terminal_event_sha256"],
                    "artifacts_before_completion_sha256": policy.canonical_json_sha256(
                        artifact_bindings
                    ),
                    "trainer_metrics_trusted": False,
                    "raw_inputs_and_checkpoints_reopened": True,
                }

        return IndependentReconstructor


    def verify(
        claim_fd: int,
        parent_fd: int,
        directory_name: str,
        manifest_file_sha256: str,
        manifest_content_sha256: str,
        expected_source_sha256: str,
    ) -> dict[str, Any]:
        reservation, reservation_raw, identity = _load_reservation(
            claim_fd,
            parent_fd,
            directory_name,
            manifest_file_sha256,
            manifest_content_sha256,
        )
        preflight, _preflight_raw = _load_preflight_first(reservation)
        if reservation["required_exact_bindings"]["exact_verifier_source_sha256"] != expected_source_sha256:
            raise PermissionError("exact verifier source binding changed")
        manifest_raw = _read_repo(policy.EXACT_EXECUTION_MANIFEST_RELATIVE_PATH)
        _assert_hash(manifest_raw, manifest_file_sha256, name="exact manifest in verifier")
        manifest = policy.validate_execution_manifest(
            _parse_json(manifest_raw, name="exact manifest in verifier"),
            require_ready=True,
        )
        if manifest["content_sha256"] != manifest_content_sha256:
            raise PermissionError("exact manifest semantic hash changed")
        artifact_raw, artifact_values = _artifact_values(claim_fd)
        _validate_static_artifacts(
            reservation,
            preflight,
            artifact_raw,
            artifact_values,
        )
        _validate_traces(artifact_values)
        backend_class = _load_fixed_backend()
        reconstruction = backend_class(
            reservation,
            artifact_raw,
            artifact_values,
        ).reconstruct()
        artifacts_before_completion_sha256 = reconstruction.pop(
            "artifacts_before_completion_sha256"
        )
        core = {
            "schema": "lewm_go2_shared_jepa_v5_full_training_v3_verification_v1",
            "status": "independently_reconstructed_pass",
            "claim_identity": list(identity),
            "execution_manifest_file_sha256": manifest_file_sha256,
            "execution_manifest_content_sha256": manifest_content_sha256,
            "artifacts_before_completion_sha256": (
                artifacts_before_completion_sha256
            ),
            "reconstruction": reconstruction,
            "trainer_metrics_trusted": False,
            "g2_open_count": 0,
            "heldout_open_count": 0,
            "runtime_navigation_hardware_open_count": 0,
            "production_or_promotion_open_count": 0,
            "runtime_ready": False,
        }
        return policy.content_value(core)


    def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--claim-fd", required=True, type=int)
        parser.add_argument("--parent-fd", required=True, type=int)
        parser.add_argument("--expected-directory-name", required=True)
        parser.add_argument("--execution-manifest-sha256", required=True)
        parser.add_argument("--execution-manifest-content-sha256", required=True)
        parser.add_argument("--expected-source-sha256", required=True)
        args = parser.parse_args(argv)
        if (
            args.claim_fd < 0
            or args.parent_fd < 0
            or args.expected_directory_name != policy.CANONICAL_EXACT_ROOT.name
            or any(
                not policy.is_sha256(value)
                for value in (
                    args.execution_manifest_sha256,
                    args.execution_manifest_content_sha256,
                    args.expected_source_sha256,
                )
            )
        ):
            raise ValueError("exact verifier arguments are malformed")
        return args


    arguments = parse_args()
    result = verify(
        arguments.claim_fd,
        arguments.parent_fd,
        arguments.expected_directory_name,
        arguments.execution_manifest_sha256,
        arguments.execution_manifest_content_sha256,
        arguments.expected_source_sha256,
    )
    print(policy.canonical_json_bytes(result).decode("ascii"))
