#!/usr/bin/env python3
"""Isolated no-tensor preflight for the causal-temporal V1 probe."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Mapping, NoReturn, Sequence


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py"
)
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_lewm_go2_rgb_causal_temporal_perception_v1_launcher_contract",
    _CONTRACT_PATH,
)
if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise ImportError("cannot load causal-temporal perception contract")
contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(contract)

RUNNER_PATH = ROOT / contract.RUNNER_RELATIVE_PATH
PREFLIGHT_ENVIRONMENT_KEY = "LEWM_CAUSAL_TEMPORAL_PERCEPTION_V1_PREFLIGHT_JSON"
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

# The child imports Torch only to inspect availability and device properties.
# It never constructs a Tensor and has no repository path or payload import.
NO_TENSOR_PREFLIGHT_PROGRAM = r"""
import json
import os
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
print(json.dumps({
    "preflight_child_process_id": os.getpid(),
    "visible_device_count": count,
    "visible_device_index": 0,
    "visible_device_name": name,
    "total_memory_bytes": memory,
    "torch_version": str(torch.__version__),
    "hip_version": str(torch.version.hip),
    "tensor_allocation_count": 0,
    "payload_open_count": 0,
    "torch_device_api_call_count": 3
}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False))
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


def _load_authority_before_hardware(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Validate exact source/review/authorization bytes before any GPU query."""
    sources = contract.current_source_bindings(ROOT)
    review_raw = _read_regular(
        ROOT / contract.REVIEW_RELATIVE_PATH,
        expected_sha256=args.review_sha256,
    )
    review = contract.validate_review(
        contract.parse_canonical_json(
            review_raw, name="launcher source review"
        ),
        expected_sources=sources,
    )
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization_raw = _read_regular(
        ROOT / contract.AUTHORIZATION_RELATIVE_PATH,
        expected_sha256=args.authorization_sha256,
    )
    authorization = contract.validate_authorization(
        contract.parse_canonical_json(
            authorization_raw, name="launcher execution authorization"
        ),
        review_binding=review_binding,
        reviewer=str(review["reviewer"]),
    )
    authorization_binding = contract.artifact_binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization_raw,
        content_sha256=str(authorization["content_sha256"]),
    )
    return {
        "source_binding_count": len(sources),
        "source_bindings_sha256": contract.canonical_json_sha256(sources),
        "source_review": review_binding,
        "execution_authorization": authorization_binding,
        "generated_runtime_input_open_count": 0,
        "torch_imported": False,
    }


def _run_no_tensor_preflight(environment: Mapping[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            NO_TENSOR_PREFLIGHT_PROGRAM,
        ],
        cwd=ROOT,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"isolated no-tensor preflight failed: {message}")
    if completed.stderr:
        raise RuntimeError("isolated no-tensor preflight wrote stderr")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("isolated no-tensor preflight was not JSON") from error
    fields = {
        "preflight_child_process_id",
        "visible_device_count",
        "visible_device_index",
        "visible_device_name",
        "total_memory_bytes",
        "torch_version",
        "hip_version",
        "tensor_allocation_count",
        "payload_open_count",
        "torch_device_api_call_count",
    }
    if type(value) is not dict or set(value) != fields:
        raise RuntimeError("isolated no-tensor preflight fields changed")
    return value


def _preflight_receipt(
    observation: Mapping[str, object],
    source_authority: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    launcher_raw = Path(__file__).resolve().read_bytes()
    core = {
        "schema": f"{contract.SCHEMA_PREFIX}_hardware_preflight_v1",
        "status": "PASS_EXACTLY_ONE_VISIBLE_DISCRETE_R9700",
        "launcher_process_id": os.getpid(),
        "source_authority": dict(source_authority),
        **dict(observation),
        "launcher_source_sha256": hashlib.sha256(launcher_raw).hexdigest(),
        "immediate_exec_required": True,
        "intervening_gpu_query_count": 0,
    }
    value = contract.with_content_sha256(core)
    return value, contract.canonical_json_bytes(value) + b"\n"


def _exec_runner(
    args: argparse.Namespace,
    *,
    receipt: Mapping[str, object],
    receipt_raw: bytes,
    environment: Mapping[str, str],
) -> NoReturn:
    if not receipt_raw.endswith(b"\n"):
        raise RuntimeError("preflight receipt is not one complete line")
    child_environment = dict(environment)
    child_environment[PREFLIGHT_ENVIRONMENT_KEY] = (
        receipt_raw[:-1].decode("ascii")
    )
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
    # This is deliberately the first and only action after receipt creation.
    # In particular, no GPU-management or Torch query may intervene.
    os.execve(sys.executable, argv, child_environment)
    raise AssertionError("os.execve unexpectedly returned")


def _launch(args: argparse.Namespace, environment: Mapping[str, str]) -> NoReturn:
    source_authority = _load_authority_before_hardware(args)
    observation = _run_no_tensor_preflight(environment)
    receipt, receipt_raw = _preflight_receipt(observation, source_authority)
    return _exec_runner(
        args,
        receipt=receipt,
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
        parser.error("both review and authorization SHA-256 values are required")
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
        raise AssertionError("isolated os.execve unexpectedly returned")
    _launch(parse_args(raw_argv), environment)
    raise AssertionError("runner os.execve unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
