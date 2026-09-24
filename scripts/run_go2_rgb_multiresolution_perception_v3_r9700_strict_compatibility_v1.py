#!/usr/bin/env python3
"""Run one reviewed synthetic R9700 strict-kernel compatibility attempt.

The parent process is standard-library-only.  Torch is imported only inside
the two isolated ``-I -B -c`` children, and only after this runner has
durably reserved its dedicated compatibility root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_lewm_go2_rgb_multiresolution_perception_v3_r9700_compat_runner_contract",
    CONTRACT_PATH,
)
if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise ImportError("cannot load strict compatibility contract")
contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(contract)

PREFLIGHT_ENVIRONMENT_KEY = "LEWM_V3_R9700_STRICT_COMPATIBILITY_PREFLIGHT_JSON"
ALLOWED_TERMINAL_NAMES = frozenset(
    {
        "reservation.json",
        "access.json",
        "result.json",
        "completed.json",
        "failed.json",
    }
)


class _PostReservationInitializationError(RuntimeError):
    """Carry live handles when the fixed root exists but setup failed."""

    def __init__(
        self,
        *,
        error: BaseException,
        root_fd: int,
        parent_fd: int,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.root_fd = root_fd
        self.parent_fd = parent_fd


def _program_literal(value: object) -> str:
    return repr(value)


_GRID_TEMPLATE = r'''
import hashlib
import json
import os
import sys
import warnings
import torch
import torch.nn.functional as F

SCHEMA = __SCHEMA__
OPERATION_HASH = __OPERATION_HASH__
SPEC = __SPEC__
EXPECTED_DETERMINISM = __DETERMINISM__

def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

def self_hashed(core):
    return {
        **core,
        "content_sha256": hashlib.sha256(canonical(core)).hexdigest(),
    }

def python_identity():
    return {
        "implementation": str(sys.implementation.name),
        "version": str(sys.version),
        "cache_tag": str(sys.implementation.cache_tag),
        "executable": str(sys.executable),
        "isolated": bool(sys.flags.isolated),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
    }

def stack_identity():
    return {
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "unknown")),
        "hip_version": str(torch.version.hip),
    }

def device_identity():
    properties = torch.cuda.get_device_properties(0)
    return {
        "visible_device_count": int(torch.cuda.device_count()),
        "visible_device_index": 0,
        "visible_device_name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
    }

def warning_rows(observed):
    rows = []
    for item in observed:
        message = str(item.message)
        rows.append({
            "category": str(item.category.__name__),
            "message": message,
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        })
    return rows

if not torch.cuda.is_available() or int(torch.cuda.device_count()) != 1:
    raise SystemExit("grid child requires exactly one visible accelerator")

stage = "strict_configuration"
exception = None
checks = {
    "strict_state_verified_before_allocation": False,
    "forward_completed": False,
    "output_finite": False,
    "backward_invoked": False,
    "backward_completed": False,
    "input_gradient_finite": False,
    "cuda_synchronize_completed": False,
    "exact_grid_call_count": False,
}
counts = {
    "grid_sample_forward_invocation_count": 0,
    "grid_sample_forward_completion_count": 0,
    "backward_invocation_count": 0,
    "backward_completion_count": 0,
    "cuda_synchronize_count": 0,
    "synthetic_dense_tensor_count": 0,
    "synthetic_grid_tensor_count": 0,
    "model_instantiation_count": 0,
    "optimizer_step_count": 0,
    "payload_open_count": 0,
}
observed_warnings = []
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        actual_determinism = {
            "requested":
                "torch.use_deterministic_algorithms(True, warn_only=False)",
            "algorithms_enabled":
                bool(torch.are_deterministic_algorithms_enabled()),
            "warn_only_enabled":
                bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "warning_count_required": 0,
            "fallback_authorized": False,
            "state_change_after_enable_authorized": False,
        }
        if actual_determinism != EXPECTED_DETERMINISM:
            raise RuntimeError("strict deterministic state did not match contract")
        checks["strict_state_verified_before_allocation"] = True

        stage = "grid_synthetic_allocation"
        dense_count = 4 * 36 * 112 * 112
        dense = torch.arange(
            dense_count,
            dtype=torch.float32,
            device="cuda:0",
        ).reshape(4, 36, 112, 112)
        dense = (dense.remainder(257.0) / 128.0 - 1.0).requires_grad_(True)
        counts["synthetic_dense_tensor_count"] = 1

        query = torch.arange(
            81920,
            dtype=torch.float32,
            device="cuda:0",
        )
        x = query.remainder(224.0) / 112.0 - 1.0
        y = torch.div(query, 224.0, rounding_mode="floor").remainder(168.0)
        y = y / 84.0 - 1.0
        padding = query.remainder(97.0) == 0.0
        x = torch.where(padding, torch.full_like(x, 2.0), x)
        y = torch.where(padding, torch.full_like(y, 2.0), y)
        grid = torch.stack((x, y), dim=-1)
        grid = grid[None].expand(4, -1, -1).contiguous()
        grid = grid.reshape(4, 128, 128, 5, 2)
        counts["synthetic_grid_tensor_count"] = 1

        stage = "grid_forward"
        flat_grid = grid.reshape(4, 81920, 2)
        sampled_chunks = []
        for start in range(0, 81920, 4096):
            counts["grid_sample_forward_invocation_count"] += 1
            sampled = F.grid_sample(
                dense,
                flat_grid[:, start:start + 4096, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            counts["grid_sample_forward_completion_count"] += 1
            sampled_chunks.append(sampled.squeeze(-1).transpose(1, 2))
        sampled_flat = torch.cat(sampled_chunks, dim=1)
        sampled_all = sampled_flat.reshape(4, 128, 128, 5, 36)
        checks["forward_completed"] = True
        checks["exact_grid_call_count"] = (
            counts["grid_sample_forward_invocation_count"] == 20
            and counts["grid_sample_forward_completion_count"] == 20
        )
        checks["output_finite"] = bool(torch.isfinite(sampled_all).all().item())
        if not checks["output_finite"]:
            raise RuntimeError("grid synthetic output was non-finite")

        scalar = sampled_all.square().mean()
        stage = "grid_backward"
        counts["backward_invocation_count"] = 1
        checks["backward_invoked"] = True
        scalar.backward()
        counts["backward_completion_count"] = 1
        checks["backward_completed"] = True
        checks["input_gradient_finite"] = bool(
            dense.grad is not None and torch.isfinite(dense.grad).all().item()
        )
        if not checks["input_gradient_finite"]:
            raise RuntimeError("grid synthetic input gradient was non-finite")

        stage = "grid_synchronize"
        torch.cuda.synchronize(0)
        counts["cuda_synchronize_count"] = 1
        checks["cuda_synchronize_completed"] = True
        stage = "completed"
    except Exception as error:
        message = str(error)
        exception = {
            "type": type(error).__name__,
            "message": message,
            "message_sha256": hashlib.sha256(
                message.encode("utf-8")
            ).hexdigest(),
        }
    observed_warnings = warning_rows(captured)

if "actual_determinism" not in locals():
    actual_determinism = {
        "requested": "torch.use_deterministic_algorithms(True, warn_only=False)",
        "algorithms_enabled": False,
        "warn_only_enabled": True,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "warning_count_required": 0,
        "fallback_authorized": False,
        "state_change_after_enable_authorized": False,
    }
status = "PASS" if exception is None and not observed_warnings else (
    "WARNING" if exception is None else "EXCEPTION"
)
receipt = self_hashed({
    "schema": SCHEMA,
    "operation": "grid_sample",
    "execution_order": 1,
    "status": status,
    "stage": stage,
    "python": python_identity(),
    "stack": stack_identity(),
    "device": device_identity(),
    "determinism": actual_determinism,
    "operation_contract_sha256": OPERATION_HASH,
    "operation_spec": SPEC,
    "warnings": observed_warnings,
    "exception": exception,
    "checks": checks,
    "counts": counts,
})
print(canonical(receipt).decode("ascii"))
'''.strip()

GRID_CHILD_PROGRAM = (
    _GRID_TEMPLATE
    .replace("__SCHEMA__", _program_literal(contract.SUBPROBE_SCHEMA))
    .replace(
        "__OPERATION_HASH__",
        _program_literal(contract.OPERATION_CONTRACT_SHA256),
    )
    .replace("__SPEC__", _program_literal(contract.GRID_OPERATION))
    .replace(
        "__DETERMINISM__",
        _program_literal(contract.DETERMINISM_CONTRACT),
    )
)

_SCATTER_TEMPLATE = r'''
import hashlib
import json
import sys
import warnings
import torch

SCHEMA = __SCHEMA__
OPERATION_HASH = __OPERATION_HASH__
SPEC = __SPEC__
EXPECTED_DETERMINISM = __DETERMINISM__

def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

def self_hashed(core):
    return {
        **core,
        "content_sha256": hashlib.sha256(canonical(core)).hexdigest(),
    }

def python_identity():
    return {
        "implementation": str(sys.implementation.name),
        "version": str(sys.version),
        "cache_tag": str(sys.implementation.cache_tag),
        "executable": str(sys.executable),
        "isolated": bool(sys.flags.isolated),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
    }

def stack_identity():
    return {
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "unknown")),
        "hip_version": str(torch.version.hip),
    }

def device_identity():
    properties = torch.cuda.get_device_properties(0)
    return {
        "visible_device_count": int(torch.cuda.device_count()),
        "visible_device_index": 0,
        "visible_device_name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
    }

def warning_rows(observed):
    rows = []
    for item in observed:
        message = str(item.message)
        rows.append({
            "category": str(item.category.__name__),
            "message": message,
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        })
    return rows

if not torch.cuda.is_available() or int(torch.cuda.device_count()) != 1:
    raise SystemExit("scatter child requires exactly one visible accelerator")

stage = "strict_configuration"
exception = None
checks = {
    "strict_state_verified_before_allocation": False,
    "all_chunks_completed": False,
    "output_finite": False,
    "backward_invoked": False,
    "backward_completed": False,
    "source_gradients_finite": False,
    "cuda_synchronize_completed": False,
    "exact_scatter_add_call_count": False,
}
counts = {
    "full_chunk_invocation_count": 0,
    "full_chunk_completion_count": 0,
    "tail_chunk_invocation_count": 0,
    "tail_chunk_completion_count": 0,
    "scatter_add_invocation_count": 0,
    "scatter_add_completion_count": 0,
    "backward_invocation_count": 0,
    "backward_completion_count": 0,
    "cuda_synchronize_count": 0,
    "synthetic_source_tensor_count": 0,
    "model_instantiation_count": 0,
    "optimizer_step_count": 0,
    "payload_open_count": 0,
}
observed_warnings = []
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        actual_determinism = {
            "requested":
                "torch.use_deterministic_algorithms(True, warn_only=False)",
            "algorithms_enabled":
                bool(torch.are_deterministic_algorithms_enabled()),
            "warn_only_enabled":
                bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "warning_count_required": 0,
            "fallback_authorized": False,
            "state_change_after_enable_authorized": False,
        }
        if actual_determinism != EXPECTED_DETERMINISM:
            raise RuntimeError("strict deterministic state did not match contract")
        checks["strict_state_verified_before_allocation"] = True

        source_leaves = []
        chunk_scalars = []
        chunk_plan = [(256, "full", chunk) for chunk in range(36)]
        chunk_plan.append((192, "tail", 0))
        for local_rays, kind, chunk_index in chunk_plan:
            if kind == "full":
                counts["full_chunk_invocation_count"] += 1
            else:
                counts["tail_chunk_invocation_count"] += 1
            stage = f"scatter_{kind}_allocation_{chunk_index}"
            destination = torch.zeros(
                4 * local_rays * 4096,
                dtype=torch.float32,
                device="cuda:0",
            )
            source = torch.full(
                (4, 64, local_rays),
                1.0 / 1024.0,
                dtype=torch.float32,
                device="cuda:0",
                requires_grad=True,
            )
            source_leaves.append(source)
            counts["synthetic_source_tensor_count"] += 1
            batch_index = torch.arange(
                4,
                dtype=torch.int64,
                device="cuda:0",
            )[:, None, None]
            depth_index = torch.arange(
                64,
                dtype=torch.int64,
                device="cuda:0",
            )[None, :, None]
            ray_index = torch.arange(
                local_rays,
                dtype=torch.int64,
                device="cuda:0",
            )[None, None, :]
            ray_group = (batch_index * local_rays + ray_index) * 4096
            base_cell = (
                ray_index * 17
                + torch.div(depth_index, 2, rounding_mode="floor") * 29
                + chunk_index * 7
            ).remainder(4096)
            valid = torch.ones(
                (4, 64, local_rays),
                dtype=torch.bool,
                device="cuda:0",
            )
            for candidate in range(4):
                stage = f"scatter_{kind}_forward_candidate_{candidate}"
                index = ray_group + (base_cell + candidate * 67).remainder(4096)
                contribution = source * ((candidate + 1.0) / 4.0)
                counts["scatter_add_invocation_count"] += 1
                destination = destination.scatter_add(
                    0,
                    index[valid],
                    contribution[valid],
                )
                counts["scatter_add_completion_count"] += 1
            chunk_scalars.append(destination.square().mean())
            if kind == "full":
                counts["full_chunk_completion_count"] += 1
            else:
                counts["tail_chunk_completion_count"] += 1

        checks["all_chunks_completed"] = (
            counts["full_chunk_completion_count"] == 36
            and counts["tail_chunk_completion_count"] == 1
        )
        checks["exact_scatter_add_call_count"] = (
            counts["scatter_add_invocation_count"] == 148
            and counts["scatter_add_completion_count"] == 148
        )
        scalar = torch.stack(chunk_scalars).mean()
        checks["output_finite"] = bool(torch.isfinite(scalar).item())
        if not checks["output_finite"]:
            raise RuntimeError("scatter synthetic output was non-finite")

        stage = "scatter_backward"
        counts["backward_invocation_count"] = 1
        checks["backward_invoked"] = True
        scalar.backward()
        counts["backward_completion_count"] = 1
        checks["backward_completed"] = True
        checks["source_gradients_finite"] = all(
            leaf.grad is not None and bool(torch.isfinite(leaf.grad).all().item())
            for leaf in source_leaves
        )
        if not checks["source_gradients_finite"]:
            raise RuntimeError("scatter synthetic source gradient was non-finite")

        stage = "scatter_synchronize"
        torch.cuda.synchronize(0)
        counts["cuda_synchronize_count"] = 1
        checks["cuda_synchronize_completed"] = True
        stage = "completed"
    except Exception as error:
        message = str(error)
        exception = {
            "type": type(error).__name__,
            "message": message,
            "message_sha256": hashlib.sha256(
                message.encode("utf-8")
            ).hexdigest(),
        }
    observed_warnings = warning_rows(captured)

if "actual_determinism" not in locals():
    actual_determinism = {
        "requested": "torch.use_deterministic_algorithms(True, warn_only=False)",
        "algorithms_enabled": False,
        "warn_only_enabled": True,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "warning_count_required": 0,
        "fallback_authorized": False,
        "state_change_after_enable_authorized": False,
    }
status = "PASS" if exception is None and not observed_warnings else (
    "WARNING" if exception is None else "EXCEPTION"
)
receipt = self_hashed({
    "schema": SCHEMA,
    "operation": "scatter_add",
    "execution_order": 2,
    "status": status,
    "stage": stage,
    "python": python_identity(),
    "stack": stack_identity(),
    "device": device_identity(),
    "determinism": actual_determinism,
    "operation_contract_sha256": OPERATION_HASH,
    "operation_spec": SPEC,
    "warnings": observed_warnings,
    "exception": exception,
    "checks": checks,
    "counts": counts,
})
print(canonical(receipt).decode("ascii"))
'''.strip()

SCATTER_CHILD_PROGRAM = (
    _SCATTER_TEMPLATE
    .replace("__SCHEMA__", _program_literal(contract.SUBPROBE_SCHEMA))
    .replace(
        "__OPERATION_HASH__",
        _program_literal(contract.OPERATION_CONTRACT_SHA256),
    )
    .replace("__SPEC__", _program_literal(contract.SCATTER_OPERATION))
    .replace(
        "__DETERMINISM__",
        _program_literal(contract.DETERMINISM_CONTRACT),
    )
)


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular(path: Path, *, expected_sha256: str) -> bytes:
    if not contract.is_sha256(expected_sha256):
        raise ValueError("expected file SHA-256 is invalid")
    if path.is_symlink():
        raise PermissionError(f"symlink authority input forbidden: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"authority input is not regular: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        _fingerprint(before) != _fingerprint(after)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise PermissionError(f"authority input changed while read: {path}")
    return raw


def _load_source_authority(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> tuple[dict[str, object], dict[str, str]]:
    if "torch" in sys.modules:
        raise PermissionError("Torch was imported in the parent runner")
    sources = contract.current_source_bindings(ROOT)
    sources_sha256 = contract.source_bindings_sha256(sources)

    decision_raw = _read_regular(
        ROOT / contract.DECISION_RELATIVE_PATH,
        expected_sha256=contract.DECISION_BINDING["file_sha256"],
    )
    decision_binding = contract.artifact_binding(
        contract.DECISION_RELATIVE_PATH,
        decision_raw,
    )
    if decision_binding != contract.DECISION_BINDING:
        raise PermissionError("committed recovery decision changed")

    preregistration_raw = _read_regular(
        ROOT / contract.PREREGISTRATION_RELATIVE_PATH,
        expected_sha256=contract.PREREGISTRATION_BINDING["file_sha256"],
    )
    preregistration = contract.validate_preregistration(
        contract.parse_canonical_json(
            preregistration_raw,
            name="V3 preregistration",
        )
    )
    preregistration_binding = contract.artifact_binding(
        contract.PREREGISTRATION_RELATIVE_PATH,
        preregistration_raw,
        content_sha256=preregistration["content_sha256"],
    )
    preregistration_binding["commit"] = contract.PREREGISTRATION_BINDING["commit"]
    if preregistration_binding != contract.PREREGISTRATION_BINDING:
        raise PermissionError("committed V3 preregistration changed")

    review_raw = _read_regular(
        ROOT / contract.REVIEW_RELATIVE_PATH,
        expected_sha256=review_file_sha256,
    )
    review = contract.validate_review(
        contract.parse_canonical_json(review_raw, name="checker source review"),
        expected_sources=sources,
        preregistration_binding=preregistration_binding,
        decision_binding=decision_binding,
    )
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )

    authorization_raw = _read_regular(
        ROOT / contract.AUTHORIZATION_RELATIVE_PATH,
        expected_sha256=authorization_file_sha256,
    )
    authorization = contract.validate_authorization(
        contract.parse_canonical_json(
            authorization_raw,
            name="checker execution authorization",
        ),
        review_binding=review_binding,
        reviewer=review["reviewer"],
        expected_source_bindings_sha256=sources_sha256,
    )
    authorization_binding = contract.artifact_binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization_raw,
        content_sha256=authorization["content_sha256"],
    )
    receipt = {
        "source_binding_count": len(sources),
        "source_bindings_sha256": sources_sha256,
        "preregistration": preregistration_binding,
        "decision": decision_binding,
        "source_review": review_binding,
        "execution_authorization": authorization_binding,
        "generated_runtime_input_open_count": 0,
        "model_or_runtime_root_open_count": 0,
        "torch_imported": False,
    }
    contract.validate_source_authority_receipt(receipt)
    return receipt, sources


def _validate_preflight_from_environment(
    *,
    expected_sha256: str,
    expected_source_authority: Mapping[str, object],
    expected_launcher_sha256: str,
) -> tuple[dict[str, object], bytes]:
    encoded = os.environ.get(PREFLIGHT_ENVIRONMENT_KEY)
    if encoded is None or "\n" in encoded or "\r" in encoded:
        raise PermissionError("exact preflight environment receipt is absent")
    try:
        raw = encoded.encode("ascii") + b"\n"
    except UnicodeEncodeError as error:
        raise PermissionError("preflight receipt is not ASCII") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError("preflight receipt file hash changed")
    value = contract.validate_preflight(
        contract.parse_canonical_json(raw, name="hardware preflight"),
        expected_source_authority=expected_source_authority,
    )
    if (
        value["launcher_process_id"] != os.getpid()
        or value["launcher_source_sha256"] != expected_launcher_sha256
        or os.environ.get("HIP_VISIBLE_DEVICES") != "0"
        or not sys.flags.isolated
        or not sys.dont_write_bytecode
    ):
        raise PermissionError("preflight immediate-exec identity changed")
    return value, raw


def _reserve_output_root() -> tuple[Path, int, int]:
    """Create only the fixed compatibility root; never inspect the V3 root."""

    relative = Path(contract.OUTPUT_ROOT_RELATIVE_PATH)
    if relative.parent.as_posix() != ".generated" or len(relative.parts) != 2:
        raise AssertionError("compatibility output root layout changed")
    parent = ROOT / ".generated"
    if parent.is_symlink():
        raise PermissionError(".generated parent may not be a symlink")
    parent_stat = parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise PermissionError(".generated parent is not a directory")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    root_fd: int | None = None
    created = False
    try:
        opened_parent = os.fstat(parent_fd)
        if (
            int(parent_stat.st_dev),
            int(parent_stat.st_ino),
            int(parent_stat.st_mode),
        ) != (
            int(opened_parent.st_dev),
            int(opened_parent.st_ino),
            int(opened_parent.st_mode),
        ):
            raise PermissionError(".generated parent changed while opened")
        os.mkdir(relative.name, 0o700, dir_fd=parent_fd)
        created = True
        root_fd = os.open(
            relative.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise PermissionError("reserved compatibility root is not a directory")
    except BaseException as error:
        if not created:
            os.close(parent_fd)
            raise
        if root_fd is None:
            try:
                root_fd = os.open(
                    relative.name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
            except BaseException as recovery_error:
                os.close(parent_fd)
                raise RuntimeError(
                    "reserved compatibility root could not be reopened for "
                    "terminalization"
                ) from recovery_error
        raise _PostReservationInitializationError(
            error=error,
            root_fd=root_fd,
            parent_fd=parent_fd,
        ) from error
    assert root_fd is not None
    return ROOT / relative, root_fd, parent_fd


def _publish_json(
    root_fd: int,
    name: str,
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    if name not in ALLOWED_TERMINAL_NAMES or "/" in name:
        raise PermissionError("terminal artifact name is not allowlisted")
    raw = contract.canonical_json_bytes(dict(value)) + b"\n"
    descriptor = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=root_fd,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short terminal artifact write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(root_fd)
    return (
        contract.artifact_binding(
            name,
            raw,
            content_sha256=str(value["content_sha256"]),
        ),
        raw,
    )


def _seal_terminal(root_fd: int, parent_fd: int) -> None:
    entries = sorted(os.listdir(root_fd))
    if not entries or any(name not in ALLOWED_TERMINAL_NAMES for name in entries):
        raise PermissionError("terminal inventory contains an unexpected entry")
    for name in entries:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_fd,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PermissionError("terminal entry is not a regular file")
            os.fchmod(descriptor, 0o444)
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o444:
                raise PermissionError("terminal file sealing failed")
        finally:
            os.close(descriptor)
    os.fsync(root_fd)
    os.fchmod(root_fd, 0o555)
    if stat.S_IMODE(os.fstat(root_fd).st_mode) != 0o555:
        raise PermissionError("terminal directory sealing failed")
    os.fsync(parent_fd)


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop(PREFLIGHT_ENVIRONMENT_KEY, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HIP_VISIBLE_DEVICES"] = "0"
    return environment


def _run_subprobe(
    *,
    program: str,
    expected_operation: str,
    preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if "torch" in sys.modules:
        raise PermissionError("Torch entered the parent runner")
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd="/tmp",
        env=_child_environment(),
        check=False,
        capture_output=True,
        text=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).decode(
            "utf-8",
            errors="replace",
        ).strip()
        raise RuntimeError(f"{expected_operation} child failed: {detail}")
    if completed.stderr:
        raise RuntimeError(f"{expected_operation} child wrote stderr")
    receipt = contract.parse_canonical_json(
        completed.stdout,
        name=f"{expected_operation} child receipt",
    )
    validated, outcome = contract.validate_subprobe_receipt(
        receipt,
        expected_operation=expected_operation,
        expected_python=preflight["python"],
        expected_stack=preflight["stack"],
        expected_device=preflight["device"],
    )
    if "torch" in sys.modules:
        raise PermissionError("subprobe imported Torch into the parent runner")
    return validated, outcome


def _reservation_receipt(
    *,
    source_authority: Mapping[str, Any],
    preflight: Mapping[str, Any],
    preflight_raw: bytes,
    attempt_identity: str,
) -> dict[str, Any]:
    return contract.with_content_sha256(
        {
            "schema": contract.RESERVATION_SCHEMA,
            "status": contract.RESERVATION_STATUS,
            "attempt_identity": attempt_identity,
            "attempt_index": contract.ATTEMPT_INDEX,
            "maximum_attempts": contract.MAXIMUM_ATTEMPTS,
            "retry_authorized": False,
            "output_root": contract.OUTPUT_ROOT_RELATIVE_PATH,
            "output_root_absent_before_reservation": True,
            "root_mode": "0700",
            "source_authority": dict(source_authority),
            "preflight": contract.artifact_binding(
                "environment:preflight",
                preflight_raw,
                content_sha256=preflight["content_sha256"],
            ),
            "operation_contract_sha256":
                contract.OPERATION_CONTRACT_SHA256,
            "output_contract_sha256": contract.OUTPUT_CONTRACT_SHA256,
            "synthetic_tensors_only": True,
            "v3_probe_root": {
                "path": contract.V3_PROBE_ROOT_RELATIVE_PATH,
                "inspected": False,
                "reserved": False,
            },
            "prohibited_open_counts": dict(contract.PROHIBITED_OPEN_COUNTS),
            "training_counts": dict(contract.ZERO_TRAINING_COUNTS),
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        }
    )


def _access_receipt(
    *,
    attempt_identity: str,
    reservation_binding: Mapping[str, Any],
    grid: Mapping[str, Any],
    scatter: Mapping[str, Any],
) -> dict[str, Any]:
    return contract.with_content_sha256(
        {
            "schema": contract.ACCESS_SCHEMA,
            "status": "ZERO_PROHIBITED_INPUT_ACCESS_SYNTHETIC_CHILDREN_ONLY",
            "attempt_identity": attempt_identity,
            "reservation": dict(reservation_binding),
            "child_process_count": 2,
            "child_order": ["grid_sample", "scatter_add"],
            "synthetic_grid_call_count":
                grid["counts"]["grid_sample_forward_invocation_count"],
            "synthetic_scatter_add_call_count":
                scatter["counts"]["scatter_add_invocation_count"],
            "prohibited_open_counts": dict(contract.PROHIBITED_OPEN_COUNTS),
            "training_counts": dict(contract.ZERO_TRAINING_COUNTS),
            "v3_probe_root": {
                "path": contract.V3_PROBE_ROOT_RELATIVE_PATH,
                "inspected": False,
                "reserved": False,
            },
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        }
    )


def _result_receipt(
    *,
    attempt_identity: str,
    reservation_binding: Mapping[str, Any],
    access_binding: Mapping[str, Any],
    preflight: Mapping[str, Any],
    grid: Mapping[str, Any],
    grid_outcome: str,
    scatter: Mapping[str, Any],
    scatter_outcome: str,
) -> dict[str, Any]:
    outcomes = {
        "grid_sample": grid_outcome,
        "scatter_add": scatter_outcome,
    }
    status = (
        contract.RESULT_PASS
        if set(outcomes.values()) == {"PASS"}
        else contract.RESULT_COMPATIBILITY_FAIL
    )
    result = contract.with_content_sha256(
        {
            "schema": contract.RESULT_SCHEMA,
            "status": status,
            "attempt_identity": attempt_identity,
            "reservation": dict(reservation_binding),
            "access": dict(access_binding),
            "python": dict(preflight["python"]),
            "stack": dict(preflight["stack"]),
            "device": dict(preflight["device"]),
            "determinism": dict(contract.DETERMINISM_CONTRACT),
            "operation_contract_sha256":
                contract.OPERATION_CONTRACT_SHA256,
            "output_contract_sha256": contract.OUTPUT_CONTRACT_SHA256,
            "subprobe_outcomes": outcomes,
            "subprobes": {
                "grid_sample": dict(grid),
                "scatter_add": dict(scatter),
            },
            "prohibited_open_counts": dict(contract.PROHIBITED_OPEN_COUNTS),
            "training_counts": dict(contract.ZERO_TRAINING_COUNTS),
            "scientific_metric": None,
            "checkpoint_qualified": False,
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        }
    )
    contract.validate_result_receipt(result)
    return result


def _completion_receipt(
    *,
    attempt_identity: str,
    reservation_binding: Mapping[str, Any],
    access_binding: Mapping[str, Any],
    result_binding: Mapping[str, Any],
    result_status: str,
) -> dict[str, Any]:
    status = (
        contract.COMPLETION_PASS
        if result_status == contract.RESULT_PASS
        else contract.COMPLETION_COMPATIBILITY_FAIL
    )
    completion = contract.with_content_sha256(
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": status,
            "attempt_identity": attempt_identity,
            "attempt_index": contract.ATTEMPT_INDEX,
            "maximum_attempts": contract.MAXIMUM_ATTEMPTS,
            "attempt_consumed": True,
            "retry_authorized": False,
            "reservation": dict(reservation_binding),
            "access": dict(access_binding),
            "result": dict(result_binding),
            "terminal_inventory": [
                "access.json",
                "completed.json",
                "reservation.json",
                "result.json",
            ],
            "terminal_file_mode": "0444",
            "terminal_root_mode": "0555",
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        }
    )
    contract.validate_completion_receipt(completion)
    return completion


def _failure_receipt(
    *,
    attempt_identity: str,
    stage: str,
    error: BaseException,
    durable_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    message = str(error)
    failure = contract.with_content_sha256(
        {
            "schema": contract.FAILURE_SCHEMA,
            "status": contract.FAILURE_STATUS,
            "attempt_identity": attempt_identity,
            "attempt_index": contract.ATTEMPT_INDEX,
            "maximum_attempts": contract.MAXIMUM_ATTEMPTS,
            "attempt_consumed": True,
            "retry_authorized": False,
            "stage": stage,
            "error": {
                "type": type(error).__name__,
                "message": message,
                "message_sha256":
                    hashlib.sha256(message.encode("utf-8")).hexdigest(),
            },
            "durable_prefix": [
                dict(durable_bindings[name])
                for name in sorted(durable_bindings)
            ],
            "compatibility_result": None,
            "prohibited_open_counts": dict(contract.PROHIBITED_OPEN_COUNTS),
            "training_counts": dict(contract.ZERO_TRAINING_COUNTS),
            "v3_probe_root": {
                "path": contract.V3_PROBE_ROOT_RELATIVE_PATH,
                "inspected": False,
                "reserved": False,
            },
            "downstream_denials": dict(contract.DOWNSTREAM_DENIALS),
        }
    )
    contract.validate_failure_receipt(failure)
    return failure


def _terminalize_failure(
    *,
    root_fd: int,
    parent_fd: int,
    attempt_identity: str,
    stage: str,
    error: BaseException,
    durable_bindings: dict[str, dict[str, Any]],
) -> int:
    terminal_error: BaseException | None = None
    try:
        failure = _failure_receipt(
            attempt_identity=attempt_identity,
            stage=stage,
            error=error,
            durable_bindings=durable_bindings,
        )
        binding, _ = _publish_json(root_fd, "failed.json", failure)
        durable_bindings["failed.json"] = binding
    except BaseException as failure_error:
        terminal_error = failure_error
    try:
        _seal_terminal(root_fd, parent_fd)
    except BaseException as seal_error:
        if terminal_error is None:
            terminal_error = seal_error
    if terminal_error is not None:
        raise RuntimeError(
            "compatibility attempt and terminalization both failed"
        ) from terminal_error
    return contract.EXIT_OPERATIONAL_FAILURE


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
    preflight_file_sha256: str,
) -> int:
    source_authority, sources = _load_source_authority(
        review_file_sha256=review_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
    )
    preflight, preflight_raw = _validate_preflight_from_environment(
        expected_sha256=preflight_file_sha256,
        expected_source_authority=source_authority,
        expected_launcher_sha256=sources[contract.LAUNCHER_RELATIVE_PATH],
    )
    attempt_identity = contract.make_attempt_identity(
        source_authority=source_authority,
        preflight=preflight,
    )

    # The mkdir itself is the only output-root absence check and reservation.
    # It targets the dedicated compatibility root, never the V3 probe root.
    try:
        _, root_fd, parent_fd = _reserve_output_root()
    except _PostReservationInitializationError as failure:
        try:
            return _terminalize_failure(
                root_fd=failure.root_fd,
                parent_fd=failure.parent_fd,
                attempt_identity=attempt_identity,
                stage="reservation_initialization",
                error=failure.error,
                durable_bindings={},
            )
        finally:
            os.close(failure.root_fd)
            os.close(failure.parent_fd)
    durable_bindings: dict[str, dict[str, Any]] = {}
    stage = "reservation_publication"
    try:
        reservation = _reservation_receipt(
            source_authority=source_authority,
            preflight=preflight,
            preflight_raw=preflight_raw,
            attempt_identity=attempt_identity,
        )
        reservation_binding, _ = _publish_json(
            root_fd,
            "reservation.json",
            reservation,
        )
        durable_bindings["reservation.json"] = reservation_binding

        stage = "post_reservation_source_authority_rehash"
        rehashed_authority, rehashed_sources = _load_source_authority(
            review_file_sha256=review_file_sha256,
            authorization_file_sha256=authorization_file_sha256,
        )
        if (
            rehashed_authority != source_authority
            or rehashed_sources != sources
        ):
            raise PermissionError("reviewed source or authority changed after reserve")

        stage = "grid_sample_subprobe"
        grid, grid_outcome = _run_subprobe(
            program=GRID_CHILD_PROGRAM,
            expected_operation="grid_sample",
            preflight=preflight,
        )
        stage = "scatter_add_subprobe"
        scatter, scatter_outcome = _run_subprobe(
            program=SCATTER_CHILD_PROGRAM,
            expected_operation="scatter_add",
            preflight=preflight,
        )

        stage = "access_publication"
        access = _access_receipt(
            attempt_identity=attempt_identity,
            reservation_binding=reservation_binding,
            grid=grid,
            scatter=scatter,
        )
        contract.validate_access_receipt(access)
        access_binding, _ = _publish_json(root_fd, "access.json", access)
        durable_bindings["access.json"] = access_binding

        stage = "result_publication"
        result = _result_receipt(
            attempt_identity=attempt_identity,
            reservation_binding=reservation_binding,
            access_binding=access_binding,
            preflight=preflight,
            grid=grid,
            grid_outcome=grid_outcome,
            scatter=scatter,
            scatter_outcome=scatter_outcome,
        )
        result_binding, _ = _publish_json(root_fd, "result.json", result)
        durable_bindings["result.json"] = result_binding

        stage = "completion_publication"
        completion = _completion_receipt(
            attempt_identity=attempt_identity,
            reservation_binding=reservation_binding,
            access_binding=access_binding,
            result_binding=result_binding,
            result_status=result["status"],
        )
        completion_binding, _ = _publish_json(
            root_fd,
            "completed.json",
            completion,
        )
        durable_bindings["completed.json"] = completion_binding

        stage = "terminal_sealing"
        _seal_terminal(root_fd, parent_fd)
        return (
            contract.EXIT_PASS
            if result["status"] == contract.RESULT_PASS
            else contract.EXIT_COMPATIBILITY_FAIL
        )
    except BaseException as error:
        return _terminalize_failure(
            root_fd=root_fd,
            parent_fd=parent_fd,
            attempt_identity=attempt_identity,
            stage=stage,
            error=error,
            durable_bindings=durable_bindings,
        )
    finally:
        os.close(root_fd)
        os.close(parent_fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args(argv)
    if (
        not args.run
        or not contract.is_sha256(args.review_sha256)
        or not contract.is_sha256(args.authorization_sha256)
        or not contract.is_sha256(args.preflight_sha256)
    ):
        parser.error("--run and all three exact SHA-256 arguments are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_parent(
            review_file_sha256=args.review_sha256,
            authorization_file_sha256=args.authorization_sha256,
            preflight_file_sha256=args.preflight_sha256,
        )
    except BaseException as error:
        # A pre-reservation failure has no authorized output namespace.  It is
        # terminal for this authorization and grants no retry.
        message = str(error)
        sys.stderr.write(
            f"pre-reservation operational failure, no retry authorized: "
            f"{type(error).__name__}: {message}\n"
        )
        return contract.EXIT_OPERATIONAL_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
