#!/usr/bin/env python3
"""Isolated no-tensor preflight and immediate-exec compatibility launcher."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Mapping, NoReturn, Sequence


# Prevent any repository bytecode write before the public launcher re-execs
# itself with the fully isolated ``-I -B`` contract.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_lewm_go2_rgb_multiresolution_perception_v3_r9700_compat_launcher_contract",
    CONTRACT_PATH,
)
if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise ImportError("cannot load strict compatibility contract")
contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(contract)

RUNNER_PATH = ROOT / contract.RUNNER_RELATIVE_PATH
PREFLIGHT_ENVIRONMENT_KEY = "LEWM_V3_R9700_STRICT_COMPATIBILITY_PREFLIGHT_JSON"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
CONFLICTING_ACCELERATOR_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "NVIDIA_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "ZE_AFFINITY_MASK",
)

# This child imports Torch only to inspect the runtime and device.  It creates
# no tensor and has no repository path or repository import.
NO_TENSOR_PREFLIGHT_PROGRAM = r"""
import json
import os
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("Torch reports no visible accelerator")
count = int(torch.cuda.device_count())
if count != 1:
    raise SystemExit(f"expected one visible accelerator, observed {count}")
properties = torch.cuda.get_device_properties(0)
name = str(properties.name)
memory = int(properties.total_memory)
if "r9700" not in name.casefold().replace(" ", ""):
    raise SystemExit(f"visible device is not R9700: {name}")
if memory < 32000000000:
    raise SystemExit(f"visible R9700 memory is too small: {memory}")
allocated = int(torch.cuda.memory_allocated(0))
reserved = int(torch.cuda.memory_reserved(0))
if allocated != 0 or reserved != 0:
    raise SystemExit(
        f"no-tensor preflight observed allocated={allocated} reserved={reserved}"
    )
value = {
    "preflight_child_process_id": os.getpid(),
    "python": {
        "implementation": str(sys.implementation.name),
        "version": str(sys.version),
        "cache_tag": str(sys.implementation.cache_tag),
        "executable": str(sys.executable),
        "isolated": bool(sys.flags.isolated),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
    },
    "stack": {
        "torch_version": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "unknown")),
        "hip_version": str(torch.version.hip),
    },
    "device": {
        "visible_device_count": count,
        "visible_device_index": 0,
        "visible_device_name": name,
        "total_memory_bytes": memory,
    },
    "tensor_allocation_count": 0,
    "memory_allocated_bytes": allocated,
    "memory_reserved_bytes": reserved,
    "payload_open_count": 0,
    "model_or_runtime_root_open_count": 0,
}
print(json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
))
""".strip()


def _launch_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        *CONFLICTING_ACCELERATOR_ENVIRONMENT,
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HIP_VISIBLE_DEVICES"] = "0"
    for name in THREAD_ENVIRONMENT:
        environment[name] = "1"
    environment.pop(PREFLIGHT_ENVIRONMENT_KEY, None)
    return environment


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


def _load_source_authority_before_hardware(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, str]]:
    """Validate reviewed source and authority before importing Torch."""

    if "torch" in sys.modules:
        raise PermissionError("Torch was imported before isolated preflight")
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
        expected_sha256=args.review_sha256,
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
        expected_sha256=args.authorization_sha256,
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


def _run_no_tensor_preflight(environment: Mapping[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            NO_TENSOR_PREFLIGHT_PROGRAM,
        ],
        cwd="/tmp",
        env=dict(environment),
        check=False,
        capture_output=True,
        text=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).decode(
            "utf-8",
            errors="replace",
        ).strip()
        raise RuntimeError(f"isolated no-tensor preflight failed: {detail}")
    if completed.stderr:
        raise RuntimeError("isolated no-tensor preflight wrote stderr")
    observation = contract.parse_canonical_json(
        completed.stdout,
        name="isolated no-tensor preflight observation",
    )
    expected = {
        "preflight_child_process_id",
        "python",
        "stack",
        "device",
        "tensor_allocation_count",
        "memory_allocated_bytes",
        "memory_reserved_bytes",
        "payload_open_count",
        "model_or_runtime_root_open_count",
    }
    if set(observation) != expected:
        raise RuntimeError("isolated no-tensor preflight fields changed")
    return observation


def _preflight_receipt(
    observation: Mapping[str, object],
    source_authority: Mapping[str, object],
    sources: Mapping[str, str],
) -> tuple[dict[str, object], bytes]:
    core = {
        "schema": contract.PREFLIGHT_SCHEMA,
        "status": contract.PREFLIGHT_STATUS,
        "launcher_process_id": os.getpid(),
        **dict(observation),
        "source_authority": dict(source_authority),
        "launcher_source_sha256": sources[contract.LAUNCHER_RELATIVE_PATH],
        "immediate_exec_required": True,
        "intervening_gpu_query_count": 0,
    }
    value = contract.with_content_sha256(core)
    contract.validate_preflight(
        value,
        expected_source_authority=source_authority,
    )
    return value, contract.canonical_json_bytes(value) + b"\n"


def _exec_runner(
    args: argparse.Namespace,
    *,
    receipt_raw: bytes,
    environment: Mapping[str, str],
) -> NoReturn:
    if not receipt_raw.endswith(b"\n") or receipt_raw.count(b"\n") != 1:
        raise RuntimeError("preflight receipt is not one complete line")
    child_environment = dict(environment)
    child_environment[PREFLIGHT_ENVIRONMENT_KEY] = receipt_raw[:-1].decode("ascii")
    argv = [
        sys.executable,
        "-I",
        "-B",
        str(RUNNER_PATH),
        "--run",
        "--review-sha256",
        args.review_sha256,
        "--authorization-sha256",
        args.authorization_sha256,
        "--preflight-sha256",
        hashlib.sha256(receipt_raw).hexdigest(),
    ]
    # The exec is the first and only action after receipt creation.
    os.execve(sys.executable, argv, child_environment)
    raise AssertionError("runner os.execve unexpectedly returned")


def _launch(args: argparse.Namespace, environment: Mapping[str, str]) -> NoReturn:
    source_authority, sources = _load_source_authority_before_hardware(args)
    observation = _run_no_tensor_preflight(environment)
    _, receipt_raw = _preflight_receipt(
        observation,
        source_authority,
        sources,
    )
    return _exec_runner(
        args,
        receipt_raw=receipt_raw,
        environment=environment,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    args = parser.parse_args(argv)
    if (
        not contract.is_sha256(args.review_sha256)
        or not contract.is_sha256(args.authorization_sha256)
    ):
        parser.error("both exact review and authorization SHA-256 values are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    environment = _launch_environment()
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        os.execve(
            sys.executable,
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                *raw_argv,
            ],
            environment,
        )
        raise AssertionError("isolated launcher os.execve unexpectedly returned")
    _launch(parse_args(raw_argv), environment)
    raise AssertionError("runner os.execve unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
