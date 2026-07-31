#!/usr/bin/env python3
"""Fail-closed placeholder for the counterfactual pilot receipt join.

The producer and receipt checker are sufficient for the one-state source
integration smoke.  A train/eval pilot join is deliberately unavailable until
a reviewed receipt-only analyzer derives its physical-rank tolerances and
FREEZE/STOP verdict from the complete 160-branch calibration collection.

In particular, this program must not accept caller-authored tolerance numbers
as if they were calibration evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn, Sequence


PILOT_JOIN_BLOCKER = (
    "pilot join is unsupported until a bound receipt-only analyzer derives "
    "repeatability tolerances and the FREEZE_PILOT_CONTRACT/STOP_SOURCE_REDESIGN "
    "decision from the exact 160-branch calibration collection"
)


class PilotJoinBlocked(RuntimeError):
    """Raised unconditionally while calibration analysis is unimplemented."""


def join_pilot(**_unused: object) -> NoReturn:
    """Refuse to mint a final pilot manifest from unanalysed calibration data."""

    raise PilotJoinBlocked(PILOT_JOIN_BLOCKER)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--expected-collection-sha256", required=True)
    parser.add_argument("--expected-collection-byte-count", required=True, type=int)
    parser.add_argument("--calibration-receipt", required=True, type=Path)
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument("--expected-calibration-byte-count", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise PilotJoinBlocked(PILOT_JOIN_BLOCKER)


if __name__ == "__main__":
    raise SystemExit(main())
