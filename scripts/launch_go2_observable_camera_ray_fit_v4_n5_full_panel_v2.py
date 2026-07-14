#!/usr/bin/env python3
"""One-shot launcher for the reviewed V4 N5 full-panel V2 successor."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as policy,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--rgb-workers", type=int, default=5, choices=range(1, 6))
    args = parser.parse_args(argv)
    if args.source_review.resolve() != policy.CANONICAL_SOURCE_REVIEW_PATH:
        raise PermissionError("N5 full-panel V2 source review path is not canonical")
    if not policy.is_sha256(args.source_review_sha256):
        raise ValueError("N5 full-panel V2 source review SHA-256 is malformed")
    return args


def _isolated_child(argv: Sequence[str]) -> int:
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HIP_VISIBLE_DEVICES"] = "0"
    environment.pop("HSA_OVERRIDE_GFX_VERSION", None)
    for name in policy.THREAD_ENVIRONMENT:
        environment[name] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *argv],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def _run_authorized(args: argparse.Namespace) -> int:
    authority = policy.verify_authority(
        args.source_review,
        args.source_review_sha256,
        purpose="exact_run",
        require_unclaimed_output=True,
    )
    from scripts import (  # protected import follows canonical source preflight
        train_go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as trainer,
    )

    summary = trainer.run_exact(authority, rgb_workers=int(args.rgb_workers))
    print(policy.canonical_json_bytes(summary).decode("ascii"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not sys.flags.isolated:
        return _isolated_child(raw_argv)
    return _run_authorized(parse_args(raw_argv))


if __name__ == "__main__":
    raise SystemExit(main())
