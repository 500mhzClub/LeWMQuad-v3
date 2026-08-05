#!/usr/bin/env python3
"""Run the fixed Shared-JEPA V5 raw-supervision Auditor V12."""
from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
for _name in (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
):
    os.environ[_name] = ""
os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
del _name

import argparse
import json
from typing import Sequence

from lewm.datasets.go2_shared_jepa_v5_raw_supervision_auditor_v12 import (
    execute_exact_audit_v12,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization-sha256",
        required=True,
        help="Frozen canonical V12 authorization file SHA-256.",
    )
    parser.add_argument(
        "--workers",
        required=True,
        type=int,
        help="Spawn worker count; the auditor accepts exact integers from 1 to 6.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = execute_exact_audit_v12(
        authorization_sha256=args.authorization_sha256,
        workers=args.workers,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
