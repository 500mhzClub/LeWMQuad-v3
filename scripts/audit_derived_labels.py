#!/usr/bin/env python3
"""Audit derived-label outputs for the mass-datagen pipeline.

Walks every ``summary.json`` under one or more label roots and checks
that ``derive_raw_rollout_labels.py`` produced a complete, joined label
stream:

* schema is ``lewm_derived_labels_v0``
* ``label_count > 0``
* ``pose_join.missing_command_count == 0``
* ``pose_join.missing_episode_info_count == 0``
* the matching ``labels.jsonl`` file exists and is non-empty

Exits non-zero on any failure. Use as the gate after
``derive_raw_rollout_labels.py`` in the mass-datagen driver.
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
        help="One or more directories containing derived-label summary.json files.",
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
            if data.get("schema") != "lewm_derived_labels_v0":
                continue
            audited += 1
            rel = str(summary.parent)
            issues: list[str] = []
            label_count = int(data.get("label_count", 0))
            if label_count <= 0:
                issues.append(f"label_count={label_count}")
            pose = data.get("pose_join") or {}
            missing_cmd = int(pose.get("missing_command_count", -1))
            missing_ep = int(pose.get("missing_episode_info_count", -1))
            if missing_cmd != 0:
                issues.append(f"missing_command_count={missing_cmd}")
            if missing_ep != 0:
                issues.append(f"missing_episode_info_count={missing_ep}")
            labels_path = data.get("labels_jsonl")
            if not labels_path:
                issues.append("labels_jsonl missing from summary")
            elif not Path(labels_path).is_file():
                issues.append(f"labels_jsonl not found: {labels_path}")
            elif Path(labels_path).stat().st_size == 0:
                issues.append("labels_jsonl is empty")
            if issues:
                failures.append((rel, "; ".join(issues)))
                print(f"[FAIL] {rel}: {'; '.join(issues)}")
            else:
                print(f"[ OK ] {rel}: labels={label_count}")
    print()
    print(f"audited {audited} label sets; {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
