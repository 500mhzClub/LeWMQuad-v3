#!/usr/bin/env python3
"""Build the swept-progress development labels from the exact V4 binding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "lewm_worlds"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from lewm.benchmarks import go2_swept_progress_survival_labels_v1 as labels  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-binding", type=Path, required=True)
    args = parser.parse_args(argv)
    binding_path = args.execution_binding.absolute()
    expected = (ROOT / labels.V4_BINDING_RELATIVE_PATH).absolute()
    if binding_path != expected:
        raise PermissionError("only the exact repository V4 binding path is accepted")
    manifest = labels.build_from_v4_binding_v1(
        binding_path,
        repository_root=ROOT,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "state_count": manifest["state_count"],
                "action_row_count": manifest["action_row_count"],
                "content_sha256": manifest["content_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
