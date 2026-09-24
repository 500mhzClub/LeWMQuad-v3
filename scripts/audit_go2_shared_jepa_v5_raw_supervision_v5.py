#!/usr/bin/env python3
"""Run the fixed Shared-JEPA V5 raw-supervision Auditor V5."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from lewm.datasets.go2_shared_jepa_v5_raw_supervision_auditor_v5 import (
    execute_exact_audit_v5,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization-sha256",
        required=True,
        help="Frozen canonical V5 authorization file SHA-256.",
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
    result = execute_exact_audit_v5(
        authorization_sha256=args.authorization_sha256,
        workers=args.workers,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
