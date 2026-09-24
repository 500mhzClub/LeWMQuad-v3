#!/usr/bin/env python3
"""Run the fixed Shared-JEPA V5 raw-supervision Auditor V13."""
from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
for _name in (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
):
    os.environ[_name] = ""
os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
del _name

_FORBIDDEN_PYTHON_ENVIRONMENT = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
)
if any(name in os.environ for name in _FORBIDDEN_PYTHON_ENVIRONMENT):
    raise RuntimeError("V13 CLI refuses caller-supplied Python path state")
if os.environ.get("PYTHONNOUSERSITE") != "1":
    raise RuntimeError("V13 CLI requires PYTHONNOUSERSITE=1")

import sys

_EXPECTED_ROOT = "/home/andrewknowles/Workspace/LeWMQuad-v3"
_EXPECTED_SCRIPT_NAME = "audit_go2_shared_jepa_v5_raw_supervision_v13.py"
_LEXICAL_SCRIPT = os.path.abspath(__file__)
_RESOLVED_SCRIPT = os.path.realpath(_LEXICAL_SCRIPT)
if (
    _LEXICAL_SCRIPT != _RESOLVED_SCRIPT
    or os.path.islink(_LEXICAL_SCRIPT)
    or not os.path.isfile(_LEXICAL_SCRIPT)
    or os.path.basename(_LEXICAL_SCRIPT) != _EXPECTED_SCRIPT_NAME
):
    raise RuntimeError("V13 CLI path is not the reviewed regular file")
_SCRIPTS_DIRECTORY = os.path.dirname(_RESOLVED_SCRIPT)
_RESOLVED_ROOT = os.path.dirname(_SCRIPTS_DIRECTORY)
if (
    os.path.basename(_SCRIPTS_DIRECTORY) != "scripts"
    or _RESOLVED_ROOT != _EXPECTED_ROOT
    or os.path.realpath(_RESOLVED_ROOT) != _RESOLVED_ROOT
    or os.path.islink(_RESOLVED_ROOT)
    or not os.path.isdir(_RESOLVED_ROOT)
    or os.path.realpath(os.getcwd()) != _RESOLVED_ROOT
):
    raise RuntimeError("V13 CLI root or current working directory changed")
_LEWM_PACKAGE = os.path.join(_RESOLVED_ROOT, "lewm")
_LEWM_INIT = os.path.join(_LEWM_PACKAGE, "__init__.py")
if (
    os.path.realpath(_LEWM_PACKAGE) != _LEWM_PACKAGE
    or os.path.islink(_LEWM_PACKAGE)
    or not os.path.isdir(_LEWM_PACKAGE)
    or os.path.realpath(_LEWM_INIT) != _LEWM_INIT
    or os.path.islink(_LEWM_INIT)
    or not os.path.isfile(_LEWM_INIT)
):
    raise RuntimeError("V13 CLI root lacks the reviewed literal lewm package")

_normalized_entries: list[str] = []
for _entry in sys.path:
    if type(_entry) is not str:
        raise RuntimeError("V13 CLI sys.path contains a non-string entry")
    _entry_lexical = os.path.abspath(_entry or os.getcwd())
    _entry_resolved = os.path.realpath(_entry_lexical)
    if _entry_resolved == _RESOLVED_ROOT:
        if _entry_lexical != _RESOLVED_ROOT:
            raise RuntimeError("V13 CLI refuses a repository-root alias")
        continue
    if _entry_resolved == _SCRIPTS_DIRECTORY:
        continue
    if os.path.isdir(_entry_lexical) and (
        os.path.exists(os.path.join(_entry_lexical, "lewm"))
        or os.path.exists(os.path.join(_entry_lexical, "lewm.py"))
    ):
        raise RuntimeError("V13 CLI refuses a foreign or installed lewm package")
    _normalized_entries.append(_entry)
sys.path[:] = [_RESOLVED_ROOT, *_normalized_entries]
if sys.path[0] != _RESOLVED_ROOT or sum(
    os.path.realpath(os.path.abspath(entry or os.getcwd())) == _RESOLVED_ROOT
    for entry in sys.path
) != 1:
    raise RuntimeError("V13 CLI failed to install one exact repository root")
del _entry, _entry_lexical, _entry_resolved, _normalized_entries

import argparse
import json
from typing import Sequence

from lewm.datasets.go2_shared_jepa_v5_raw_supervision_auditor_v13 import (
    execute_exact_audit_v13,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization-sha256",
        required=True,
        help="Frozen canonical V13 authorization file SHA-256.",
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
    result = execute_exact_audit_v13(
        authorization_sha256=args.authorization_sha256,
        workers=args.workers,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
