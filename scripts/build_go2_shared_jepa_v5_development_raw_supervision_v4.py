#!/usr/bin/env python3
"""Build Shared-JEPA V5 raw supervision through the V4 authority gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "lewm_worlds") not in sys.path:
    sys.path.insert(0, str(ROOT / "lewm_worlds"))

from lewm.datasets.go2_shared_jepa_v5_raw_supervision_builder_v4 import (  # noqa: E402
    CANONICAL_OUTPUT,
    execute_exact_build_v4,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = execute_exact_build_v4(
        authorization_sha256=str(args.authorization_sha256),
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "output": str(CANONICAL_OUTPUT),
                "content_sha256": manifest["content_sha256"],
                "status": manifest["status"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
