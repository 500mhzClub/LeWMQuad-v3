#!/usr/bin/env python3
"""Run the fixed-path Shared-JEPA V5 raw-supervision V2 exact audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "lewm_worlds") not in sys.path:
    sys.path.insert(0, str(ROOT / "lewm_worlds"))

from lewm.datasets.go2_shared_jepa_v5_raw_supervision_auditor_v2 import (  # noqa: E402
    MAX_WORKERS,
    execute_exact_audit_v2,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the one canonical development raw-supervision artifact. "
            "Dataset and result paths are intentionally not configurable."
        )
    )
    parser.add_argument(
        "--manifest-sha256",
        required=True,
        help="Frozen file SHA-256 of the canonical dataset manifest.json.",
    )
    parser.add_argument(
        "--authorization-sha256",
        required=True,
        help="Frozen file SHA-256 of the fixed dual-review authorization JSON.",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = execute_exact_audit_v2(
        expected_manifest_file_sha256=str(args.manifest_sha256),
        expected_authorization_file_sha256=str(args.authorization_sha256),
        workers=args.workers,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
