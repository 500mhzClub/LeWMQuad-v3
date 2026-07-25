#!/usr/bin/env python3
"""Authority-first launcher for Local-Correspondence Transport JEPA V7."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_ACTION_CONDITIONED_LOCAL_CORRESPONDENCE_"
    "TRANSPORT_JEPA_V7_PREFLIGHT_JSON"
)


def _source_only_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    (
        "_lewm_go2_rgb_jepa_encoder_pretraining_"
        "v7_local_correspondence_transport_launcher_contract"
    ),
    ROOT / "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py",
)
_BASE = _source_only_module(
    (
        "_lewm_go2_rgb_jepa_encoder_pretraining_"
        "v7_local_correspondence_transport_base_launcher"
    ),
    ROOT / "scripts/launch_go2_rgb_causal_temporal_perception_v1.py",
)

# Reuse the reviewed source-authority validation and isolated environment.
# V7 deliberately defers the Torch-importing hardware child until after the
# runner has reserved the one-shot output root.
_BASE.contract = contract
_BASE.RUNNER_PATH = ROOT / contract.RUNNER_RELATIVE_PATH
_BASE.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
_BASE.__file__ = str(Path(__file__).resolve())

NO_TENSOR_PREFLIGHT_PROGRAM = _BASE.NO_TENSOR_PREFLIGHT_PROGRAM
parse_args = _BASE.parse_args


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    environment = _BASE._launch_environment()
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
        raise AssertionError("isolated launcher exec unexpectedly returned")
    args = parse_args(raw_argv)
    _BASE._load_authority_before_hardware(args)
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / contract.RUNNER_RELATIVE_PATH),
            "--run",
            "--review-sha256",
            args.review_sha256,
            "--authorization-sha256",
            args.authorization_sha256,
        ],
        environment,
    )
    raise AssertionError("runner exec unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
