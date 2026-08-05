#!/usr/bin/env python3
"""Independent stdlib verifier for the payload-free V5 V2 GPU preflight."""
from __future__ import annotations

import argparse
import hashlib
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

from lewm.benchmarks import go2_shared_jepa_v5_full_training_v4_policy as policy


if __name__ == "__main__":
    ROOT = SCRIPT_ROOT

    def _directory_flags() -> int:
        if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
            raise PermissionError("preflight verifier requires no-follow directories")
        return (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )


    def _read_at(directory_fd: int, name: str) -> bytes:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise PermissionError("preflight verifier leaf escaped")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise PermissionError("preflight evidence is not singly linked")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
            fingerprint = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_uid,
                value.st_gid,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if fingerprint(before) != fingerprint(after):
                raise RuntimeError("preflight evidence changed while verified")
            return b"".join(chunks)
        finally:
            os.close(descriptor)


    def _read_repo(relative: str) -> bytes:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise PermissionError("preflight verifier source path escaped")
        descriptors = [os.open(ROOT, _directory_flags())]
        parent_fd = descriptors[0]
        try:
            for component in path.parts[:-1]:
                parent_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
                descriptors.append(parent_fd)
            return _read_at(parent_fd, path.name)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


    def _parse(raw: bytes, name: str) -> dict[str, Any]:
        return policy.parse_canonical_json(raw, name=name)


    def _verify_source_closure(value: Mapping[str, Any]) -> None:
        expected_fields = {
            "schema",
            "preflight_authorization_file_sha256",
            "reviewed_sources",
            "reviewed_design_and_model_bindings",
            "payload_open_count",
            "content_sha256",
        }
        if (
            set(value) != expected_fields
            or value.get("schema")
            != "lewm_go2_shared_jepa_v5_full_training_v4_preflight_source_closure_v1"
            or value.get("reviewed_design_and_model_bindings")
            != policy.reviewed_source_bindings()
            or value.get("payload_open_count") != 0
        ):
            raise PermissionError("preflight source closure changed")
        sources = value.get("reviewed_sources")
        if not isinstance(sources, Mapping) or set(sources) != set(
            policy.IMPLEMENTATION_SOURCE_PATHS
        ):
            raise PermissionError("preflight implementation closure changed")
        if any(not policy.is_sha256(item) for item in sources.values()):
            raise ValueError("preflight implementation source hash is malformed")


    def _verify_receipt(value: Mapping[str, Any]) -> None:
        required = {
            "schema",
            "status",
            "source_bindings",
            "preflight_authorization_file_sha256",
            "device_contract",
            "environment",
            "observed_device",
            "production_model_config_sha256",
            "optimizer_contract",
            "tensor_contract",
            "loss_components",
            "loss_values",
            "gradient_norm_before_clip",
            "gradient_norm_after_clip",
            "optimizer_step_count",
            "ema_step_count",
            "accumulated_backward_count",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "terminal_synchronization_passed",
            "payload_open_count",
            "forbidden_open_count",
            "access_ledger_terminal_sha256",
            "content_sha256",
        }
        if set(value) != required:
            raise PermissionError("preflight receipt fields changed")
        if (
            value.get("schema") != policy.PREFLIGHT_RECEIPT_SCHEMA
            or value.get("status") != "PASS"
            or value.get("source_bindings") is None
            or value.get("device_contract") != policy.DEVICE_CONTRACT
            or value.get("optimizer_contract") != policy.OPTIMIZER_CONTRACT
            or value.get("optimizer_step_count") != 1
            or value.get("ema_step_count") != 1
            or value.get("accumulated_backward_count") != 4
            or value.get("terminal_synchronization_passed") is not True
            or value.get("payload_open_count") != 0
            or value.get("forbidden_open_count") != 0
            or not policy.is_sha256(value.get("production_model_config_sha256"))
            or not policy.is_sha256(value.get("access_ledger_terminal_sha256"))
        ):
            raise PermissionError("preflight receipt contract failed")
        sources = value.get("source_bindings")
        if not isinstance(sources, Mapping) or any(
            not policy.is_sha256(item) for item in sources.values()
        ):
            raise PermissionError("preflight receipt source bindings changed")
        environment = value.get("environment")
        if (
            not isinstance(environment, Mapping)
            or not policy.is_sha256(environment.get("full_environment_sha256"))
            or not isinstance(environment.get("variable_names"), list)
            or not isinstance(environment.get("accelerator"), Mapping)
            or environment["accelerator"].get("HIP_VISIBLE_DEVICES") != "0"
            or environment["accelerator"].get("ROCR_VISIBLE_DEVICES") != "0"
            or environment["accelerator"].get("HSA_OVERRIDE_GFX_VERSION") is not None
        ):
            raise PermissionError("preflight environment evidence failed")
        observed = value.get("observed_device")
        if (
            not isinstance(observed, Mapping)
            or observed.get("name") != policy.DEVICE_CONTRACT["device_name"]
            or int(observed.get("total_memory_bytes", 0))
            < policy.DEVICE_CONTRACT["minimum_total_memory_bytes"]
            or observed.get("device") != "cuda:0"
        ):
            raise PermissionError("preflight device evidence failed")
        losses = value.get("loss_values")
        expected_loss_components = {
            "joint_total",
            "established_jepa_total",
            "jepa_prediction",
            "jepa_equivariance",
            "jepa_action_contrast",
            "jepa_variance",
            "jepa_warped_persistence",
            "v4_pair_total",
            "v4_current_total",
            "v4_next_total",
            "v4_current_hierarchical_first_hit_nll",
            "v4_current_target_bin_offset_smooth_l1",
            "v4_current_ground_clear_distance_state_balanced_bce",
            "v4_current_derived_raster_hierarchical_bce",
            "v4_current_derived_raster_cell_nll",
            "v4_next_hierarchical_first_hit_nll",
            "v4_next_target_bin_offset_smooth_l1",
            "v4_next_ground_clear_distance_state_balanced_bce",
            "v4_next_derived_raster_hierarchical_bce",
            "v4_next_derived_raster_cell_nll",
        }
        tensor = value.get("tensor_contract")
        if (
            not isinstance(losses, Mapping)
            or set(losses) != expected_loss_components
            or value.get("loss_components") != sorted(expected_loss_components)
            or not isinstance(tensor, Mapping)
            or tensor
            != {
                "current_rgb": [
                    policy.MICROBATCH_SIZE,
                    3,
                    policy.MODEL_IMAGE_SIZE,
                    policy.MODEL_IMAGE_SIZE,
                ],
                "next_rgb": [
                    policy.MICROBATCH_SIZE,
                    3,
                    policy.MODEL_IMAGE_SIZE,
                    policy.MODEL_IMAGE_SIZE,
                ],
                "action": [policy.MICROBATCH_SIZE, 9],
                "delta": [policy.MICROBATCH_SIZE, 3],
                "prediction_mask": [
                    policy.MICROBATCH_SIZE,
                    *policy.MODEL_BEV_SHAPE,
                ],
                "dtype": "torch.float32",
                "device": "cuda:0",
                "source_shape": list(policy.MODEL_SOURCE_SHAPE),
                "pixel_ray_shape": list(policy.MODEL_PIXEL_RAY_SHAPE),
            }
        ):
            raise ValueError("preflight losses are missing")
        for name, number in losses.items():
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not float("-inf") < float(number) < float("inf")
            ):
                raise ValueError(f"preflight loss is nonfinite: {name}")
        for name in (
            "gradient_norm_before_clip",
            "gradient_norm_after_clip",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            number = value.get(name)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or float(number) < 0.0
            ):
                raise ValueError(f"preflight measurement is invalid: {name}")
        if float(value["gradient_norm_after_clip"]) > 1.000001:
            raise PermissionError("preflight gradient clip changed")


    def verify(
        claim_fd: int,
        parent_fd: int,
        expected_directory_name: str,
        preflight_authorization_sha256: str,
    ) -> dict[str, Any]:
        opened = os.fstat(claim_fd)
        named = os.stat(
            expected_directory_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise PermissionError("preflight claim identity changed")
        names = ("reservation.json", "source_closure.json", "access_ledger.json", "gpu_smoke_receipt.json")
        if set(os.listdir(claim_fd)) != set(names):
            raise PermissionError("preflight verifier inventory changed")
        raw = {name: _read_at(claim_fd, name) for name in names}
        values = {name: _parse(raw[name], name) for name in names}
        reservation = values["reservation.json"]
        if (
            reservation.get("schema") != policy.PREFLIGHT_RESERVATION_SCHEMA
            or reservation.get("preflight_authorization_file_sha256")
            != preflight_authorization_sha256
            or reservation.get("retry_authorized") is not False
        ):
            raise PermissionError("preflight reservation changed")
        _verify_source_closure(values["source_closure.json"])
        preflight_authorization_raw = _read_repo(
            policy.EXACT_BINDING_PREFLIGHT_AUTHORIZATION_RELATIVE_PATH
        )
        if (
            hashlib.sha256(preflight_authorization_raw).hexdigest()
            != preflight_authorization_sha256
        ):
            raise PermissionError("preflight exact-binding preflight authorization hash changed")
        preflight_authorization = policy.validate_exact_binding_preflight_authorization(
            _parse(preflight_authorization_raw, "exact-binding preflight authorization")
        )
        source_closure = values["source_closure.json"]
        if (
            source_closure.get("preflight_authorization_file_sha256")
            != preflight_authorization_sha256
            or source_closure.get("reviewed_sources")
            != preflight_authorization["reviewed_sources"]
        ):
            raise PermissionError("preflight exact-binding preflight authorization closure changed")
        expected_source_events: dict[str, tuple[str, int]] = {}
        for relative, expected in {
            **policy.reviewed_source_bindings(),
            **dict(preflight_authorization["reviewed_sources"]),
        }.items():
            source_raw = _read_repo(relative)
            if hashlib.sha256(source_raw).hexdigest() != expected:
                raise PermissionError(f"preflight reviewed source changed: {relative}")
            expected_source_events[relative] = (expected, len(source_raw))
        ledger_value = values["access_ledger.json"]
        if ledger_value.get("schema") != policy.ACCESS_LEDGER_SCHEMA:
            raise PermissionError("preflight ledger schema changed")
        events = ledger_value.get("events")
        if not isinstance(events, list):
            raise ValueError("preflight ledger events are missing")
        ledger = policy.validate_access_ledger(events)
        if (
            ledger_value.get("summary") != ledger
            or ledger_value.get("payload_open_count") != 0
            or ledger_value.get("forbidden_open_count") != 0
        ):
            raise PermissionError("preflight ledger summary changed")
        observed_source_events: dict[str, tuple[str, int]] = {}
        for event in events:
            relative = event.get("relative_path")
            if not isinstance(relative, str) or relative in observed_source_events:
                raise PermissionError("preflight source ledger path is duplicated")
            if (
                event.get("stage") != "preflight_source_closure"
                or event.get("arm") is not None
                or event.get("role") != "source_closure"
                or event.get("operation") != "read_and_rehash"
                or event.get("expected_sha256") != event.get("observed_sha256")
                or isinstance(event.get("byte_count"), bool)
                or not isinstance(event.get("byte_count"), int)
            ):
                raise PermissionError("preflight source ledger event changed")
            observed_source_events[relative] = (
                str(event["observed_sha256"]),
                int(event["byte_count"]),
            )
        if observed_source_events != expected_source_events:
            raise PermissionError(
                "preflight source ledger did not exactly cover reviewed sources"
            )
        _verify_receipt(values["gpu_smoke_receipt.json"])
        receipt = values["gpu_smoke_receipt.json"]
        expected_receipt_sources = {
            **policy.reviewed_source_bindings(),
            **dict(values["source_closure.json"]["reviewed_sources"]),
        }
        if (
            receipt.get("preflight_authorization_file_sha256")
            != preflight_authorization_sha256
            or receipt.get("source_bindings") != expected_receipt_sources
            or receipt.get("access_ledger_terminal_sha256")
            != ledger["terminal_event_sha256"]
        ):
            raise PermissionError("preflight receipt does not bind the ledger")
        bindings = {
            name: policy.artifact_binding(
                name,
                raw[name],
                content_sha256=str(values[name]["content_sha256"]),
            )
            for name in names
        }
        core = {
            "schema": "lewm_go2_shared_jepa_v5_full_training_v4_preflight_verification_v1",
            "status": "independently_reconstructed_pass",
            "preflight_authorization_file_sha256": preflight_authorization_sha256,
            "claim_identity": [opened.st_dev, opened.st_ino],
            "artifacts": bindings,
            "payload_open_count": 0,
            "forbidden_open_count": 0,
        }
        return policy.content_value(core)


    def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--claim-fd", required=True, type=int)
        parser.add_argument("--parent-fd", required=True, type=int)
        parser.add_argument("--expected-directory-name", required=True)
        parser.add_argument("--preflight-authorization-sha256", required=True)
        args = parser.parse_args(argv)
        if (
            args.claim_fd < 0
            or args.parent_fd < 0
            or args.expected_directory_name != policy.CANONICAL_PREFLIGHT_ROOT.name
            or not policy.is_sha256(args.preflight_authorization_sha256)
        ):
            raise ValueError("preflight verifier arguments are malformed")
        return args


    arguments = parse_args()
    summary = verify(
        arguments.claim_fd,
        arguments.parent_fd,
        arguments.expected_directory_name,
        arguments.preflight_authorization_sha256,
    )
    print(policy.canonical_json_bytes(summary).decode("ascii"))
