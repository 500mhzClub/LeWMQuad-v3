#!/usr/bin/env python3
"""One-shot payload-free GPU0 preflight for Shared JEPA V5 training V2."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lewm.benchmarks import go2_shared_jepa_v5_full_training_v2_policy as policy


if __name__ == "__main__":
    ROOT = SCRIPT_ROOT

    def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )


    def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_gid,
        )


    def _directory_flags() -> int:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        if not nofollow or not directory:
            raise PermissionError("preflight requires no-follow directory opens")
        return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


    @dataclass
    class DirectoryEntry:
        parent_fd: int
        name: str
        child_fd: int
        fingerprint: tuple[int, ...]


    @dataclass
    class DirectoryChain:
        anchor_fd: int
        anchor_fingerprint: tuple[int, ...]
        descriptors: list[int]
        entries: list[DirectoryEntry]
        path_fds: dict[Path, int]
        closed: bool = False


    @dataclass(frozen=True)
    class PreflightReservation:
        directory_fd: int
        directory_identity: tuple[int, int]
        parent_fd: int
        directory_name: str
        chain: DirectoryChain
        reservation_value: Mapping[str, Any]
        reservation_raw: bytes
        implementation_review: Mapping[str, Any]
        implementation_review_file_sha256: str
        reviewed_source_bytes: Mapping[str, bytes]


    def _assert_chain(chain: DirectoryChain) -> None:
        if chain.closed:
            raise PermissionError("preflight directory chain is closed")
        if _directory_identity(os.fstat(chain.anchor_fd)) != _directory_identity_from_fingerprint(chain.anchor_fingerprint):
            raise PermissionError("preflight filesystem-root descriptor changed")
        for entry in chain.entries:
            named = os.stat(entry.name, dir_fd=entry.parent_fd, follow_symlinks=False)
            opened = os.fstat(entry.child_fd)
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or _directory_identity(named) != _directory_identity_from_fingerprint(entry.fingerprint)
                or _directory_identity(opened) != _directory_identity_from_fingerprint(entry.fingerprint)
            ):
                raise PermissionError("preflight canonical ancestry changed")


    def _refresh_chain(chain: DirectoryChain, mutable_fds: set[int]) -> None:
        if chain.closed:
            raise PermissionError("preflight directory chain is closed")
        if _directory_identity(os.fstat(chain.anchor_fd)) != _directory_identity_from_fingerprint(chain.anchor_fingerprint):
            raise PermissionError("preflight filesystem-root descriptor changed")
        for entry in chain.entries:
            named = os.stat(entry.name, dir_fd=entry.parent_fd, follow_symlinks=False)
            opened = os.fstat(entry.child_fd)
            named_fingerprint = _fingerprint(named)
            opened_fingerprint = _fingerprint(opened)
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or _directory_identity(named) != _directory_identity_from_fingerprint(entry.fingerprint)
                or _directory_identity(opened) != _directory_identity_from_fingerprint(entry.fingerprint)
            ):
                raise PermissionError("preflight canonical ancestry changed")
            if entry.child_fd in mutable_fds:
                entry.fingerprint = opened_fingerprint


    def _directory_identity_from_fingerprint(value: tuple[int, ...]) -> tuple[int, ...]:
        return (value[0], value[1], value[2], value[4], value[5])


    def _open_parent_chain(final_parent: Path) -> DirectoryChain:
        final_parent = Path(final_parent)
        if (
            not final_parent.is_absolute()
            or not final_parent.is_relative_to(ROOT)
            or any(part in {"", ".", ".."} for part in final_parent.parts[1:])
        ):
            raise PermissionError("preflight parent escaped the repository")
        filesystem_root = Path(final_parent.anchor)
        anchor_before = filesystem_root.stat(follow_symlinks=False)
        anchor_fd = os.open(filesystem_root, _directory_flags())
        chain = DirectoryChain(
            anchor_fd=anchor_fd,
            anchor_fingerprint=_fingerprint(anchor_before),
            descriptors=[anchor_fd],
            entries=[],
            path_fds={filesystem_root: anchor_fd},
        )
        try:
            if _directory_identity(os.fstat(anchor_fd)) != _directory_identity_from_fingerprint(chain.anchor_fingerprint):
                raise PermissionError("preflight filesystem root changed during open")
            parent_fd = anchor_fd
            current = filesystem_root
            repository_depth = len(ROOT.parts) - 1
            for index, component in enumerate(final_parent.parts[1:]):
                created = False
                try:
                    before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    if index < repository_depth:
                        raise PermissionError("repository component is missing")
                    os.mkdir(component, 0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    created = True
                    before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise PermissionError("preflight parent component is not a directory")
                child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
                chain.descriptors.append(child_fd)
                fingerprint = _fingerprint(before)
                if _directory_identity(os.fstat(child_fd)) != _directory_identity_from_fingerprint(fingerprint):
                    raise PermissionError("preflight parent component changed")
                if created and chain.entries:
                    _refresh_chain(chain, {parent_fd})
                chain.entries.append(DirectoryEntry(parent_fd, component, child_fd, fingerprint))
                current = current / component
                chain.path_fds[current] = child_fd
                parent_fd = child_fd
            _assert_chain(chain)
            return chain
        except BaseException:
            _close_chain(chain)
            raise


    def _close_chain(chain: DirectoryChain) -> None:
        if chain.closed:
            return
        chain.closed = True
        for descriptor in reversed(chain.descriptors):
            os.close(descriptor)


    def _read_relative_source(relative: str) -> bytes:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise PermissionError("reviewed source path escaped")
        root_fd = os.open(ROOT, _directory_flags())
        descriptors = [root_fd]
        parent_fd = root_fd
        try:
            for component in path.parts[:-1]:
                descriptor = os.open(component, _directory_flags(), dir_fd=parent_fd)
                descriptors.append(descriptor)
                parent_fd = descriptor
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise PermissionError("reviewed source is not singly linked")
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                if _fingerprint(before) != _fingerprint(after):
                    raise RuntimeError("reviewed source changed while read")
                return b"".join(chunks)
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


    def _load_implementation_review(expected_file_sha256: str) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
        if not policy.is_sha256(expected_file_sha256):
            raise ValueError("implementation review SHA-256 is malformed")
        raw = _read_relative_source(policy.IMPLEMENTATION_REVIEW_RELATIVE_PATH)
        if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
            raise PermissionError("implementation review file hash changed")
        review = policy.parse_canonical_json(raw, name="implementation review")
        policy.validate_implementation_review(review)
        reviewed: dict[str, bytes] = {}
        for relative, expected in review["reviewed_sources"].items():
            source_raw = _read_relative_source(relative)
            if hashlib.sha256(source_raw).hexdigest() != expected:
                raise PermissionError(f"implementation source changed: {relative}")
            reviewed[relative] = source_raw
        for relative, expected in policy.reviewed_source_bindings().items():
            source_raw = _read_relative_source(relative)
            if hashlib.sha256(source_raw).hexdigest() != expected:
                raise PermissionError(f"reviewed parent changed: {relative}")
            reviewed[relative] = source_raw
        return review, raw, reviewed


    def _assert_claim(reservation: PreflightReservation) -> None:
        _assert_chain(reservation.chain)
        opened = os.fstat(reservation.directory_fd)
        named = os.stat(
            reservation.directory_name,
            dir_fd=reservation.parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (opened.st_dev, opened.st_ino) != reservation.directory_identity
            or (named.st_dev, named.st_ino) != reservation.directory_identity
        ):
            raise PermissionError("preflight claim identity changed")


    def _write_exclusive(reservation: PreflightReservation, name: str, raw: bytes) -> None:
        _assert_claim(reservation)
        if Path(name).name != name or name in {"", ".", ".."}:
            raise PermissionError("preflight artifact path escaped")
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=reservation.directory_fd,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.fsync(reservation.directory_fd)
        _assert_claim(reservation)


    def _reserve_preflight(
        review: Mapping[str, Any],
        review_raw: bytes,
        review_file_sha256: str,
        reviewed_source_bytes: Mapping[str, bytes],
    ) -> PreflightReservation:
        del review_raw
        final = policy.CANONICAL_PREFLIGHT_ROOT
        chain = _open_parent_chain(final.parent)
        parent_fd = chain.path_fds[final.parent]
        directory_fd = -1
        try:
            os.mkdir(final.name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            _refresh_chain(chain, {parent_fd})
            directory_fd = os.open(final.name, _directory_flags(), dir_fd=parent_fd)
            metadata = os.fstat(directory_fd)
            identity = (metadata.st_dev, metadata.st_ino)
            reservation_core = {
                "schema": policy.PREFLIGHT_RESERVATION_SCHEMA,
                "status": "reserved_before_gpu_runtime_access",
                "namespace": policy.PREFLIGHT_ROOT_RELATIVE_PATH,
                "directory_identity": list(identity),
                "implementation_review_file_sha256": review_file_sha256,
                "implementation_review_content_sha256": review["content_sha256"],
                "payload_access_authorized": False,
                "exact_attempt_created": False,
                "retry_authorized": False,
            }
            reservation_value = policy.content_value(reservation_core)
            reservation_raw = policy.canonical_json_bytes(reservation_value) + b"\n"
            reservation = PreflightReservation(
                directory_fd=directory_fd,
                directory_identity=identity,
                parent_fd=parent_fd,
                directory_name=final.name,
                chain=chain,
                reservation_value=reservation_value,
                reservation_raw=reservation_raw,
                implementation_review=review,
                implementation_review_file_sha256=review_file_sha256,
                reviewed_source_bytes=dict(reviewed_source_bytes),
            )
            _write_exclusive(reservation, "reservation.json", reservation_raw)
            return reservation
        except BaseException:
            if directory_fd >= 0:
                os.close(directory_fd)
            _close_chain(chain)
            raise


    def _synthetic_supervision(model_module: Any, pair: Any, torch: Any) -> tuple[Any, Any]:
        def frame_supervision(frame: Any) -> Any:
            hazard = frame.evidence.pixel_first_hit_hazard_logits
            pixel_shape = (hazard.shape[0], hazard.shape[2], hazard.shape[3])
            hit = torch.zeros(pixel_shape, dtype=torch.bool, device=hazard.device)
            hit[:, 0, 0] = True
            distance = torch.zeros(pixel_shape, dtype=hazard.dtype, device=hazard.device)
            distance[hit] = model_module.DEPTH_NEAR_EDGE_M + 0.25 * model_module.DEPTH_BIN_SIZE_M
            in_frustum = frame.evidence.ground_query_in_frustum.detach().clone()
            parity = torch.arange(in_frustum.numel(), device=in_frustum.device).reshape(in_frustum.shape) % 2 == 0
            clear = in_frustum & parity
            labels = torch.zeros(
                (hazard.shape[0], *model_module.OUTPUT_SHAPE),
                dtype=torch.long,
                device=hazard.device,
            )
            labels[:, 12:40, 16:48] = 1
            labels[:, 28:32, 30:34] = 2
            return model_module.ObservableCameraRayV4FrameSupervisionV5(
                pixel_hit_mask=hit,
                pixel_first_hit_distance_m=distance,
                ground_support_in_frustum=in_frustum,
                ground_support_clear_to_target=clear,
                target_raster_labels=labels,
            )

        return frame_supervision(pair.current), frame_supervision(pair.next)


    def _load_production_backend() -> Any:
        if (
            os.environ.get("HIP_VISIBLE_DEVICES") != "0"
            or os.environ.get("ROCR_VISIBLE_DEVICES") != "0"
            or "HSA_OVERRIDE_GFX_VERSION" in os.environ
        ):
            raise PermissionError("preflight accelerator environment changed")
        import torch
        from lewm.models import shared_observable_camera_ray_jepa_v5 as model_module

        class FixedProductionBackend:
            def run(self) -> dict[str, Any]:
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise PermissionError("preflight requires exactly one visible ROCm device")
                device = torch.device("cuda:0")
                properties = torch.cuda.get_device_properties(device)
                if (
                    properties.name != policy.DEVICE_CONTRACT["device_name"]
                    or int(properties.total_memory)
                    < policy.DEVICE_CONTRACT["minimum_total_memory_bytes"]
                ):
                    raise PermissionError("preflight R9700 identity or memory changed")
                torch.manual_seed(policy.INITIALIZATION_SEED)
                torch.cuda.manual_seed_all(policy.INITIALIZATION_SEED)
                torch.use_deterministic_algorithms(True)
                torch.cuda.reset_peak_memory_stats(device)
                config = model_module.SharedObservableCameraRayJepaV5Config()
                model = model_module.SharedObservableCameraRayJepaV5(config).to(device).train()
                optimizer = torch.optim.AdamW(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    lr=1e-4,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=1e-4,
                    amsgrad=False,
                )
                optimizer.zero_grad(set_to_none=True)
                accumulated: dict[str, float] = {}
                tensor_contract: dict[str, Any] | None = None
                for microbatch_index in range(policy.ACCUMULATION_STEPS):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(policy.INITIALIZATION_SEED + microbatch_index)
                    current = torch.randn(
                        policy.MICROBATCH_SIZE,
                        3,
                        config.image_size,
                        config.image_size,
                        generator=generator,
                        device=device,
                        dtype=torch.float32,
                    )
                    next_image = torch.randn(
                        current.shape,
                        generator=generator,
                        device=device,
                        dtype=torch.float32,
                    )
                    action = torch.zeros(policy.MICROBATCH_SIZE, config.action_dim, device=device)
                    action[:, microbatch_index % config.action_dim] = 1.0
                    wrong_action = torch.roll(action, shifts=1, dims=1)
                    realized = torch.zeros(policy.MICROBATCH_SIZE, 3, device=device)
                    realized[:, 0] = 0.05
                    commanded = torch.zeros_like(realized)
                    commanded[:, 0] = 0.10
                    wrong_delta = -commanded
                    wrong_action_delta = torch.zeros_like(realized)
                    wrong_action_delta[:, 1] = 0.10
                    origin = torch.tensor((0.326, 0.02, 0.043), device=device)[None].expand(policy.MICROBATCH_SIZE, -1).clone()
                    basis = torch.tensor(
                        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
                        device=device,
                    )[None].expand(policy.MICROBATCH_SIZE, -1, -1).clone()
                    ground = torch.full((policy.MICROBATCH_SIZE,), -0.35, device=device)
                    prediction_mask = torch.ones(
                        (policy.MICROBATCH_SIZE, *config.bev_size),
                        dtype=torch.bool,
                        device=device,
                    )
                    pair = model.forward_training_pair(
                        current,
                        next_image,
                        action,
                        realized,
                        commanded_delta_pose_current=commanded,
                        current_camera_origin_body_m=origin,
                        current_camera_basis_body_fru=basis,
                        current_ground_plane_z_body_m=ground,
                        next_camera_origin_body_m=origin,
                        next_camera_basis_body_fru=basis,
                        next_ground_plane_z_body_m=ground,
                        next_prediction_mask=prediction_mask,
                        diagnostic_wrong_action=wrong_action,
                        diagnostic_wrong_action_delta_pose_current=wrong_action_delta,
                        diagnostic_wrong_commanded_delta_pose_current=wrong_delta,
                    )
                    current_supervision, next_supervision = _synthetic_supervision(
                        model_module,
                        pair,
                        torch,
                    )
                    joint = model.combine_joint_losses(
                        pair,
                        current_supervision,
                        next_supervision,
                    )
                    if not bool(torch.isfinite(joint.total).item()):
                        raise FloatingPointError("preflight joint loss is nonfinite")
                    (joint.total / policy.ACCUMULATION_STEPS).backward()
                    components = {
                        "joint_total": joint.total,
                        "established_jepa_total": joint.established_jepa.total,
                        "jepa_prediction": joint.established_jepa.prediction,
                        "jepa_equivariance": joint.established_jepa.equivariance,
                        "jepa_action_contrast": joint.established_jepa.action_contrast,
                        "jepa_variance": joint.established_jepa.variance,
                        "jepa_warped_persistence": joint.established_jepa.warped_persistence,
                        "v4_pair_total": joint.observable_camera_ray_v4.total,
                        "v4_current_total": joint.observable_camera_ray_v4.current.total,
                        "v4_next_total": joint.observable_camera_ray_v4.next.total,
                        "v4_current_ordered_first_hit_nll": joint.observable_camera_ray_v4.current.ordered_first_hit_nll,
                        "v4_current_target_bin_offset_smooth_l1": joint.observable_camera_ray_v4.current.target_bin_offset_smooth_l1,
                        "v4_current_ground_clear_distance_state_balanced_bce": joint.observable_camera_ray_v4.current.ground_clear_distance_state_balanced_bce,
                        "v4_current_derived_raster_hierarchical_bce": joint.observable_camera_ray_v4.current.derived_raster_hierarchical_bce.total,
                        "v4_next_ordered_first_hit_nll": joint.observable_camera_ray_v4.next.ordered_first_hit_nll,
                        "v4_next_target_bin_offset_smooth_l1": joint.observable_camera_ray_v4.next.target_bin_offset_smooth_l1,
                        "v4_next_ground_clear_distance_state_balanced_bce": joint.observable_camera_ray_v4.next.ground_clear_distance_state_balanced_bce,
                        "v4_next_derived_raster_hierarchical_bce": joint.observable_camera_ray_v4.next.derived_raster_hierarchical_bce.total,
                    }
                    for name, value in components.items():
                        accumulated[name] = accumulated.get(name, 0.0) + float(value.detach().cpu()) / policy.ACCUMULATION_STEPS
                    tensor_contract = {
                        "current_rgb": list(current.shape),
                        "next_rgb": list(next_image.shape),
                        "action": list(action.shape),
                        "delta": list(realized.shape),
                        "prediction_mask": list(prediction_mask.shape),
                        "dtype": str(current.dtype),
                        "device": str(current.device),
                        "source_shape": list(config.source_shape),
                        "pixel_ray_shape": list(config.pixel_ray_shape),
                    }
                parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
                squared_before = sum(float(parameter.grad.detach().float().pow(2).sum().cpu()) for parameter in parameters)
                gradient_before = math.sqrt(squared_before)
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                squared_after = sum(float(parameter.grad.detach().float().pow(2).sum().cpu()) for parameter in parameters)
                gradient_after = math.sqrt(squared_after)
                optimizer.step()
                model.update_ema_target_after_optimizer_step()
                torch.cuda.synchronize(device)
                return {
                    "observed_device": {
                        "device": "cuda:0",
                        "name": properties.name,
                        "total_memory_bytes": int(properties.total_memory),
                        "multi_processor_count": int(properties.multi_processor_count),
                        "gcn_arch_name": str(getattr(properties, "gcnArchName", "")),
                        "torch_version": str(torch.__version__),
                        "hip_version": str(torch.version.hip),
                        "torch_git_version": str(torch.version.git_version),
                        "python_version": platform.python_version(),
                        "python_implementation": platform.python_implementation(),
                        "platform": platform.platform(),
                        "kernel": platform.release(),
                    },
                    "production_model_config_sha256": config.content_sha256,
                    "tensor_contract": tensor_contract,
                    "loss_components": sorted(accumulated),
                    "loss_values": accumulated,
                    "gradient_norm_before_clip": gradient_before,
                    "gradient_norm_after_clip": gradient_after,
                    "optimizer_step_count": 1,
                    "ema_step_count": 1,
                    "accumulated_backward_count": policy.ACCUMULATION_STEPS,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "terminal_synchronization_passed": True,
                }

        return FixedProductionBackend()


    def _fresh_smoke_environment() -> dict[str, str]:
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["HIP_VISIBLE_DEVICES"] = "0"
        environment["ROCR_VISIBLE_DEVICES"] = "0"
        environment.pop("HSA_OVERRIDE_GFX_VERSION", None)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
        return environment


    def _run_fixed_smoke_child(reservation: PreflightReservation) -> dict[str, Any]:
        expected_source_sha256 = reservation.implementation_review["reviewed_sources"][
            policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH
        ]
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(ROOT / policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH),
                "--fixed-smoke-child",
                "--implementation-review-sha256",
                reservation.implementation_review_file_sha256,
                "--claim-fd",
                str(reservation.directory_fd),
                "--parent-fd",
                str(reservation.parent_fd),
                "--expected-directory-name",
                reservation.directory_name,
                "--expected-source-sha256",
                str(expected_source_sha256),
            ],
            cwd=ROOT,
            env=_fresh_smoke_environment(),
            pass_fds=(reservation.directory_fd, reservation.parent_fd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError("fixed preflight smoke child failed")
        summary = policy.parse_canonical_json(
            completed.stdout,
            name="fixed preflight smoke-child summary",
        )
        if (
            summary.get("schema")
            != "lewm_go2_shared_jepa_v5_full_training_v2_smoke_child_v1"
            or summary.get("status") != "gpu_smoke_complete_process_terminating"
            or summary.get("claim_identity")
            != list(reservation.directory_identity)
            or summary.get("implementation_review_file_sha256")
            != reservation.implementation_review_file_sha256
            or summary.get("preflight_executor_source_sha256")
            != expected_source_sha256
            or summary.get("payload_open_count") != 0
            or summary.get("forbidden_open_count") != 0
            or not isinstance(summary.get("measurements"), Mapping)
        ):
            raise PermissionError("fixed preflight smoke-child summary changed")
        return _validate_backend_measurements(summary["measurements"])


    def _smoke_child_entry(arguments: argparse.Namespace) -> dict[str, Any]:
        opened = os.fstat(arguments.claim_fd)
        named = os.stat(
            arguments.expected_directory_name,
            dir_fd=arguments.parent_fd,
            follow_symlinks=False,
        )
        identity = (int(opened.st_dev), int(opened.st_ino))
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or identity != (int(named.st_dev), int(named.st_ino))
        ):
            raise PermissionError("smoke child claim identity changed")
        descriptor = os.open(
            "reservation.json",
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=arguments.claim_fd,
        )
        try:
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        reservation = policy.parse_canonical_json(
            b"".join(chunks),
            name="preflight reservation in smoke child",
        )
        if (
            reservation.get("schema") != policy.PREFLIGHT_RESERVATION_SCHEMA
            or reservation.get("status") != "reserved_before_gpu_runtime_access"
            or reservation.get("directory_identity") != list(identity)
            or reservation.get("implementation_review_file_sha256")
            != arguments.implementation_review_sha256
            or reservation.get("payload_access_authorized") is not False
            or reservation.get("retry_authorized") is not False
        ):
            raise PermissionError("smoke child reservation changed")
        source_raw = _read_relative_source(policy.PREFLIGHT_EXECUTOR_RELATIVE_PATH)
        if hashlib.sha256(source_raw).hexdigest() != arguments.expected_source_sha256:
            raise PermissionError("smoke child source binding changed")
        measurements = _validate_backend_measurements(
            _load_production_backend().run()
        )
        return policy.content_value(
            {
                "schema": "lewm_go2_shared_jepa_v5_full_training_v2_smoke_child_v1",
                "status": "gpu_smoke_complete_process_terminating",
                "claim_identity": list(identity),
                "implementation_review_file_sha256": (
                    arguments.implementation_review_sha256
                ),
                "preflight_executor_source_sha256": (
                    arguments.expected_source_sha256
                ),
                "measurements": measurements,
                "payload_open_count": 0,
                "forbidden_open_count": 0,
            }
        )


    def _validate_backend_measurements(value: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "observed_device",
            "production_model_config_sha256",
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
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise PermissionError("preflight backend measurement fields changed")
        for name in ("gradient_norm_before_clip", "gradient_norm_after_clip"):
            number = value[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                raise ValueError(f"preflight {name} is nonfinite")
        if (
            value["optimizer_step_count"] != 1
            or value["ema_step_count"] != 1
            or value["accumulated_backward_count"] != 4
            or value["terminal_synchronization_passed"] is not True
        ):
            raise PermissionError("preflight backend cadence changed")
        return dict(value)


    def _publish_preflight(
        reservation: PreflightReservation,
        measurements: Mapping[str, Any],
    ) -> dict[str, Any]:
        _assert_claim(reservation)
        source_hashes = {
            relative: hashlib.sha256(raw).hexdigest()
            for relative, raw in reservation.reviewed_source_bytes.items()
        }
        source_core = {
            "schema": "lewm_go2_shared_jepa_v5_full_training_v2_preflight_source_closure_v1",
            "implementation_review_file_sha256": reservation.implementation_review_file_sha256,
            "reviewed_sources": dict(reservation.implementation_review["reviewed_sources"]),
            "reviewed_design_and_model_bindings": policy.reviewed_source_bindings(),
            "payload_open_count": 0,
        }
        for relative, expected in {**source_core["reviewed_sources"], **source_core["reviewed_design_and_model_bindings"]}.items():
            if source_hashes.get(relative) != expected:
                raise PermissionError(f"preflight source closure changed: {relative}")
        source_value = policy.content_value(source_core)
        source_raw = policy.canonical_json_bytes(source_value) + b"\n"
        _write_exclusive(reservation, "source_closure.json", source_raw)

        events: list[dict[str, Any]] = []
        for relative, expected in sorted(source_hashes.items()):
            events.append(
                policy.append_access_event(
                    events,
                    stage="preflight_source_closure",
                    arm=None,
                    role="source_closure",
                    operation="read_and_rehash",
                    relative_path=relative,
                    expected_sha256=expected,
                    observed_sha256=expected,
                    byte_count=len(reservation.reviewed_source_bytes[relative]),
                    process_identity=str(os.getpid()),
                )
            )
        ledger_summary = policy.validate_access_ledger(events)
        ledger_value = policy.content_value(
            {
                "schema": policy.ACCESS_LEDGER_SCHEMA,
                "events": events,
                "summary": ledger_summary,
                "payload_open_count": 0,
                "forbidden_open_count": 0,
            }
        )
        ledger_raw = policy.canonical_json_bytes(ledger_value) + b"\n"
        _write_exclusive(reservation, "access_ledger.json", ledger_raw)

        receipt_core = {
            "schema": policy.PREFLIGHT_RECEIPT_SCHEMA,
            "status": "PASS",
            "source_bindings": {
                **policy.reviewed_source_bindings(),
                **dict(reservation.implementation_review["reviewed_sources"]),
            },
            "implementation_review_file_sha256": (
                reservation.implementation_review_file_sha256
            ),
            "device_contract": policy.DEVICE_CONTRACT,
            "environment": {
                "full_environment_sha256": policy.canonical_json_sha256(
                    dict(sorted(os.environ.items()))
                ),
                "variable_names": sorted(os.environ),
                "accelerator": {
                    name: os.environ.get(name)
                    for name in (
                        "HIP_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES",
                        "CUDA_VISIBLE_DEVICES",
                        "HSA_OVERRIDE_GFX_VERSION",
                    )
                },
                "thread_limits": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            **_validate_backend_measurements(measurements),
            "optimizer_contract": policy.OPTIMIZER_CONTRACT,
            "payload_open_count": 0,
            "forbidden_open_count": 0,
            "access_ledger_terminal_sha256": ledger_summary["terminal_event_sha256"],
        }
        receipt_value = policy.content_value(receipt_core)
        receipt_raw = policy.canonical_json_bytes(receipt_value) + b"\n"
        _write_exclusive(reservation, "gpu_smoke_receipt.json", receipt_raw)
        return {
            "source_closure.json": (source_value, source_raw),
            "access_ledger.json": (ledger_value, ledger_raw),
            "gpu_smoke_receipt.json": (receipt_value, receipt_raw),
        }


    def _run_independent_verifier(reservation: PreflightReservation) -> dict[str, Any]:
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        verifier = ROOT / policy.PREFLIGHT_VERIFIER_RELATIVE_PATH
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(verifier),
                "--claim-fd",
                str(reservation.directory_fd),
                "--parent-fd",
                str(reservation.parent_fd),
                "--expected-directory-name",
                reservation.directory_name,
                "--implementation-review-sha256",
                reservation.implementation_review_file_sha256,
            ],
            cwd=ROOT,
            env=environment,
            pass_fds=(reservation.directory_fd, reservation.parent_fd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError("independent preflight verification failed")
        return policy.parse_canonical_json(
            completed.stdout,
            name="independent preflight verification",
        )


    def _publish_completion(
        reservation: PreflightReservation,
        artifacts: Mapping[str, tuple[Mapping[str, Any], bytes]],
        verification: Mapping[str, Any],
    ) -> dict[str, Any]:
        inventory = {
            "reservation.json": policy.artifact_binding(
                "reservation.json",
                reservation.reservation_raw,
                content_sha256=str(reservation.reservation_value["content_sha256"]),
            )
        }
        for name, (value, raw) in artifacts.items():
            inventory[name] = policy.artifact_binding(
                name,
                raw,
                content_sha256=str(value["content_sha256"]),
            )
        expected_before_completion = set(policy.PREFLIGHT_INVENTORY[:-1])
        observed_names = set(os.listdir(reservation.directory_fd))
        if observed_names != expected_before_completion:
            raise PermissionError("preflight directory inventory changed")
        for name in observed_names:
            metadata = os.stat(
                name,
                dir_fd=reservation.directory_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PermissionError("preflight inventory contains a linked or special entry")
        if (
            verification.get("schema")
            != "lewm_go2_shared_jepa_v5_full_training_v2_preflight_verification_v1"
            or verification.get("status") != "independently_reconstructed_pass"
            or verification.get("implementation_review_file_sha256")
            != reservation.implementation_review_file_sha256
            or verification.get("claim_identity")
            != list(reservation.directory_identity)
            or verification.get("artifacts") != inventory
            or verification.get("payload_open_count") != 0
            or verification.get("forbidden_open_count") != 0
        ):
            raise PermissionError("independent preflight verification changed")
        completion_core = {
            "schema": policy.PREFLIGHT_COMPLETION_SCHEMA,
            "status": "completed_after_independent_reconstruction",
            "namespace": policy.PREFLIGHT_ROOT_RELATIVE_PATH,
            "directory_identity": list(reservation.directory_identity),
            "ordered_inventory": list(policy.PREFLIGHT_INVENTORY),
            "artifacts_before_completion": inventory,
            "independent_verification": dict(verification),
            "exact_attempt_created": False,
            "payload_open_count": 0,
            "retry_authorized": False,
            "exact_execution_authorized": False,
        }
        completion = policy.content_value(completion_core)
        raw = policy.canonical_json_bytes(completion) + b"\n"
        _write_exclusive(reservation, "completed.json", raw)
        os.fsync(reservation.directory_fd)
        os.fsync(reservation.parent_fd)
        return completion


    def _terminalize_failure(reservation: PreflightReservation, error: BaseException) -> None:
        core = {
            "schema": "lewm_go2_shared_jepa_v5_full_training_v2_preflight_failure_v1",
            "status": "terminal_failure_no_retry",
            "failure_class": type(error).__name__,
            "payload_open_count": 0,
            "exact_attempt_created": False,
            "retry_authorized": False,
        }
        raw = policy.canonical_json_bytes(policy.content_value(core)) + b"\n"
        try:
            _write_exclusive(reservation, "failed.json", raw)
        except FileExistsError:
            pass
        os.fsync(reservation.directory_fd)
        os.fsync(reservation.parent_fd)


    def _close_reservation(reservation: PreflightReservation) -> None:
        os.close(reservation.directory_fd)
        _close_chain(reservation.chain)


    def _orchestrate_preflight(
        reserve_operation: Callable[[], PreflightReservation],
        smoke_operation: Callable[[PreflightReservation], Mapping[str, Any]],
        publish_operation: Callable[[PreflightReservation, Mapping[str, Any]], Mapping[str, tuple[Mapping[str, Any], bytes]]],
        verification_operation: Callable[[PreflightReservation], Mapping[str, Any]],
        completion_operation: Callable[[PreflightReservation, Mapping[str, tuple[Mapping[str, Any], bytes]], Mapping[str, Any]], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        reservation = reserve_operation()
        try:
            measurements = smoke_operation(reservation)
            artifacts = publish_operation(reservation, measurements)
            verification = verification_operation(reservation)
            return completion_operation(reservation, artifacts, verification)
        except BaseException as error:
            _terminalize_failure(reservation, error)
            raise
        finally:
            _close_reservation(reservation)


    def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--implementation-review-sha256", required=True)
        parser.add_argument("--fixed-smoke-child", action="store_true")
        parser.add_argument("--claim-fd", type=int)
        parser.add_argument("--parent-fd", type=int)
        parser.add_argument("--expected-directory-name")
        parser.add_argument("--expected-source-sha256")
        args = parser.parse_args(argv)
        if not policy.is_sha256(args.implementation_review_sha256):
            raise ValueError("implementation review SHA-256 is malformed")
        child_values = (
            args.claim_fd,
            args.parent_fd,
            args.expected_directory_name,
            args.expected_source_sha256,
        )
        if args.fixed_smoke_child:
            if (
                any(value is None for value in child_values)
                or args.claim_fd < 0
                or args.parent_fd < 0
                or args.expected_directory_name
                != policy.CANONICAL_PREFLIGHT_ROOT.name
                or not policy.is_sha256(args.expected_source_sha256)
            ):
                raise ValueError("fixed smoke-child arguments are malformed")
        elif any(value is not None for value in child_values):
            raise PermissionError("smoke-child descriptors require the fixed child mode")
        return args


    def _isolated_child(argv: Sequence[str]) -> int:
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["HIP_VISIBLE_DEVICES"] = "0"
        environment["ROCR_VISIBLE_DEVICES"] = "0"
        environment.pop("HSA_OVERRIDE_GFX_VERSION", None)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *argv],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        return int(completed.returncode)


    raw_arguments = list(sys.argv[1:])
    if not sys.flags.isolated:
        raise SystemExit(_isolated_child(raw_arguments))
    arguments = parse_args(raw_arguments)
    if arguments.fixed_smoke_child:
        smoke_summary = _smoke_child_entry(arguments)
        print(policy.canonical_json_bytes(smoke_summary).decode("ascii"))
        raise SystemExit(0)
    implementation_review, review_raw, reviewed_sources = _load_implementation_review(
        arguments.implementation_review_sha256
    )
    completion = _orchestrate_preflight(
        lambda: _reserve_preflight(
            implementation_review,
            review_raw,
            arguments.implementation_review_sha256,
            reviewed_sources,
        ),
        _run_fixed_smoke_child,
        _publish_preflight,
        _run_independent_verifier,
        _publish_completion,
    )
    print(policy.canonical_json_bytes(completion).decode("ascii"))
