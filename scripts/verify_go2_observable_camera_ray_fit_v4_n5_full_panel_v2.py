#!/usr/bin/env python3
"""Independent exact-inference verifier for the V2 full-panel attempt."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as policy,
)


def _retained() -> Any:
    from scripts import (
        verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained,
    )

    retained.policy = policy
    return retained


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _retained().parse_args(argv)


def _validate_attempt_bundle(
    authority: policy.VerifiedAuthorityV2,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return _retained()._validate_attempt_bundle(authority, args)


def _validate_checkpoint(
    raw: bytes,
    *,
    expected_binding: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    return _retained()._validate_checkpoint(
        raw,
        expected_binding=expected_binding,
        expected_metadata=expected_metadata,
    )


def recompute_evaluation(**kwargs: Any) -> dict[str, Any]:
    return _retained().recompute_evaluation(**kwargs)


def _compute_receipt(
    authority: policy.VerifiedAuthorityV2,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    return _retained()._compute_receipt(authority, bundle)


def run(args: argparse.Namespace) -> dict[str, Any]:
    authority = policy.verify_authority(
        args.source_review,
        args.source_review_sha256,
        purpose="metric_verification",
        require_unclaimed_output=False,
    )
    policy.transition_authority(
        authority,
        purpose="metric_verification",
        target_path=policy.CANONICAL_METRIC_RECEIPT_PATH,
        from_states=("issued",),
        to_state="active",
    )
    bundle = _validate_attempt_bundle(authority, args)
    receipt = _compute_receipt(authority, bundle)
    policy.write_exclusive(policy.CANONICAL_METRIC_RECEIPT_PATH, receipt)
    return receipt


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


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not sys.flags.isolated:
        return _isolated_child(raw_argv)
    receipt = run(parse_args(raw_argv))
    print(policy.canonical_json_bytes(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
