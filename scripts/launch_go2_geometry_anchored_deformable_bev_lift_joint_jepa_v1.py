#!/usr/bin/env python3
"""Source-only authority gate and exact runtime handoff for joint-JEPA V1."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, NoReturn, Sequence


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).resolve()
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/"
    "go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1.py"
)
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


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source-only contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_module(
    "_lewm_geometry_anchored_joint_jepa_v1_launcher_contract", CONTRACT_PATH
)
RUNNER_PATH = ROOT / contract.RUNNER_RELATIVE_PATH
OUTPUT_ROOT = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
if ROOT / contract.LAUNCHER_RELATIVE_PATH != LAUNCHER_PATH:
    raise PermissionError("geometry-anchored joint-JEPA launcher path changed")


def _sha256_argument(value: str) -> str:
    if not contract.is_sha256(value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sha256", required=True, type=_sha256_argument)
    parser.add_argument(
        "--authorization-sha256", required=True, type=_sha256_argument
    )
    return parser.parse_args(argv)


def _artifact_binding(
    path: str, raw: bytes, value: dict[str, Any]
) -> dict[str, Any]:
    return contract.artifact_binding(
        path, raw, content_sha256=str(value["content_sha256"])
    )


def _read_exact(relative: str, expected_sha256: str) -> bytes:
    raw = contract._read_regular_source(ROOT / relative)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError(f"authority SHA-256 changed: {relative}")
    return raw


def _validate_authority(
    *, review_sha256: str, authorization_sha256: str
) -> dict[str, Any]:
    """Validate frozen sources, review, and authorization without runtime data."""

    sources_before = contract.current_source_bindings(ROOT)
    manifest_raw = contract._read_regular_source(
        ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH
    )
    manifest = contract.validate_source_manifest(manifest_raw, ROOT)
    manifest_binding = _artifact_binding(
        contract.SOURCE_MANIFEST_RELATIVE_PATH, manifest_raw, manifest
    )
    review_raw = _read_exact(contract.REVIEW_RELATIVE_PATH, review_sha256)
    review = contract.validate_review(review_raw, manifest_binding, root=ROOT)
    review_binding = _artifact_binding(
        contract.REVIEW_RELATIVE_PATH, review_raw, review
    )
    authorization_raw = _read_exact(
        contract.AUTHORIZATION_RELATIVE_PATH, authorization_sha256
    )
    authorization = contract.validate_authorization(
        authorization_raw, review_binding, root=ROOT
    )
    if sources_before != contract.current_source_bindings(ROOT):
        raise PermissionError("source changed while launch authority was checked")
    if (
        hashlib.sha256(
            contract._read_regular_source(ROOT / contract.REVIEW_RELATIVE_PATH)
        ).hexdigest()
        != review_sha256
        or hashlib.sha256(
            contract._read_regular_source(
                ROOT / contract.AUTHORIZATION_RELATIVE_PATH
            )
        ).hexdigest()
        != authorization_sha256
    ):
        raise PermissionError("review or authorization changed during preflight")
    return {
        "source_count": len(sources_before),
        "reviewer": review["reviewer"],
        "authorizer": authorization["authorizer"],
    }


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
    return environment


def _runtime_argv(args: argparse.Namespace) -> list[str]:
    return [
        contract.RUNTIME_INTERPRETER_PATH,
        *contract.RUNTIME_INTERPRETER_ARGUMENTS,
        str(RUNNER_PATH),
        "--review-sha256",
        args.review_sha256,
        "--authorization-sha256",
        args.authorization_sha256,
    ]


def _exec_runtime(args: argparse.Namespace) -> NoReturn:
    argv = _runtime_argv(args)
    os.execve(
        contract.RUNTIME_INTERPRETER_PATH,
        argv,
        _launch_environment(),
    )
    raise AssertionError("runtime os.execve unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_authority(
        review_sha256=args.review_sha256,
        authorization_sha256=args.authorization_sha256,
    )
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError("the one-shot output root already exists")
    _exec_runtime(args)


if __name__ == "__main__":
    raise SystemExit(main())
