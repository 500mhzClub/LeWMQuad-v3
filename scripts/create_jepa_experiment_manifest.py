#!/usr/bin/env python3
"""Create or verify a content-addressed JEPA experiment manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.experiment_manifest import (  # noqa: E402
    build_experiment_manifest,
    verify_manifest_files,
    write_json,
)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("file argument must use NAME=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id")
    parser.add_argument("--input", action="append", type=_named_path, default=[])
    parser.add_argument("--artifact", action="append", type=_named_path, default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--run-command")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()

    if args.verify is not None:
        report = verify_manifest_files(json.loads(args.verify.read_text()))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passes"] else 1

    if not args.experiment_id or args.output is None:
        parser.error("--experiment-id and --output are required unless --verify is used")
    if len(dict(args.input)) != len(args.input):
        parser.error("input names must be unique")
    if len(dict(args.artifact)) != len(args.artifact):
        parser.error("artifact names must be unique")
    config = json.loads(args.config_json.read_text()) if args.config_json else {}
    if not isinstance(config, dict):
        parser.error("--config-json must contain a JSON object")
    manifest = build_experiment_manifest(
        experiment_id=args.experiment_id,
        repository_root=REPO_ROOT,
        inputs=dict(args.input),
        artifacts=dict(args.artifact),
        config=config,
        seeds=args.seed,
        run_command=args.run_command,
    )
    write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
