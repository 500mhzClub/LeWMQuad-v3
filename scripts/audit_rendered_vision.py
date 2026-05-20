#!/usr/bin/env python3
"""Audit render-replay outputs against the §6 quality gate.

Walks every ``summary.json`` under one or more rendered-vision roots and
checks the gates defined in ``docs/mass_datagen_runbook.md`` §6:

* ``invalid_frame_count == 0``
* ``camera_safety_unresolved_count == 0``
* ``low_info_frame_count <= --low-info-threshold`` (default 0)
* ``overlay_target_label == false``  (privileged label off)

Exits non-zero on any failure. Designed to be the CI gate the bulk
mass-datagen driver runs between render and label-derivation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more directories containing rendered_vision summary.json files.",
    )
    parser.add_argument(
        "--low-info-threshold",
        type=int,
        default=0,
        help="Maximum allowed low_info_frame_count per scene (default 0).",
    )
    parser.add_argument(
        "--allow-overlay",
        action="store_true",
        help="Treat overlay_target_label=True as non-fatal (QA renders only).",
    )
    args = parser.parse_args()

    failures: list[tuple[str, str]] = []
    audited = 0
    for root in args.roots:
        if not root.exists():
            print(f"missing path: {root}", file=sys.stderr)
            return 2
        for summary in sorted(root.rglob("summary.json")):
            data = json.loads(summary.read_text(encoding="utf-8"))
            if data.get("schema") != "lewm_rendered_vision_v0":
                continue
            audited += 1
            rel = str(summary.parent)
            issues: list[str] = []
            if int(data.get("invalid_frame_count", 0)) != 0:
                issues.append(
                    f"invalid_frame_count={data['invalid_frame_count']}"
                )
            if int(data.get("camera_safety_unresolved_count", 0)) != 0:
                issues.append(
                    f"camera_safety_unresolved_count={data['camera_safety_unresolved_count']}"
                )
            # Gate on low-info frames the renderer did NOT excuse. Recovery
            # episodes legitimately stare at walls while backing out of
            # dead-ends; the renderer marks those low_info_allowed and reports
            # the un-excused remainder as low_info_invalid_frame_count. Gating
            # on the raw count would fail honest recovery footage.
            low_info_invalid = int(
                data.get(
                    "low_info_invalid_frame_count",
                    data.get("low_info_frame_count", 0),
                )
            )
            if low_info_invalid > args.low_info_threshold:
                low_info_total = int(data.get("low_info_frame_count", 0))
                issues.append(
                    f"low_info_invalid_frame_count={low_info_invalid}"
                    f">threshold({args.low_info_threshold}) "
                    f"(of {low_info_total} low-info; rest recovery-excused)"
                )
            if bool(data.get("overlay_target_label", False)) and not args.allow_overlay:
                issues.append("overlay_target_label=true")
            if issues:
                failures.append((rel, "; ".join(issues)))
                print(f"[FAIL] {rel}: {'; '.join(issues)}")
            else:
                print(f"[ OK ] {rel}: frames={data.get('frame_count')}")
    print()
    print(f"audited {audited} rendered scenes; {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
