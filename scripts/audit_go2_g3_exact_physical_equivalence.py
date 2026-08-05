#!/usr/bin/env python3
"""Launch the captured G3 exact-physical runner in a fresh sealed process."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


CANONICAL_ROOT = Path("/home/andrewknowles/Workspace/LeWMQuad-v3").resolve()
RUNNER_RELATIVE_PATH = Path(
    "lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v1.py"
)
EXPECTED_RUNNER_SOURCE_SHA256 = (
    "4fbceaa49519d811de3f1508c99099c8b1ddda8cb7dacefcd8aa153a05f4a3b3"
)
DEFAULT_OUTPUT = (
    CANONICAL_ROOT
    / ".generated/go2_g3_exact_physical_equivalence/v1/candidate.json"
)
THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_SEALED_MODULE_NAME = "_lewm_captured_g3_exact_physical_runner_v1"
_CHILD_BOOTSTRAP = r"""
import base64, hashlib, json, pathlib, sys, types
request = json.loads(sys.stdin.read())
payload = base64.b64decode(request["runner_source_b64"], validate=True)
expected = request["runner_source_sha256"]
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError("captured G3 runner bytes changed in transit")
module = types.ModuleType(request["module_name"])
module.__file__ = request["runner_path"]
module.__package__ = ""
sys.modules[request["module_name"]] = module
exec(compile(payload, request["runner_path"], "exec"), module.__dict__)
if request["operation"] == "probe":
    result = module._sealed_bootstrap_probe(expected)
elif request["operation"] == "run":
    result = module._sealed_run(
        output=pathlib.Path(request["output"]),
        workers=request["workers"],
        expected_runner_source_sha256=expected,
    )
else:
    raise RuntimeError("unsupported sealed G3 runner operation")
print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
"""


def _read_fixed_runner_source() -> bytes:
    expected_wrapper = CANONICAL_ROOT / "scripts/audit_go2_g3_exact_physical_equivalence.py"
    if Path(__file__).resolve() != expected_wrapper:
        raise PermissionError("G3 audit launcher was copied or imported from another root")
    path = CANONICAL_ROOT / RUNNER_RELATIVE_PATH
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PermissionError("captured G3 runner is not a singly-linked regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != EXPECTED_RUNNER_SOURCE_SHA256:
        raise PermissionError("captured G3 runner source hash is not frozen")
    return payload


def _invoke_sealed_child(
    *,
    operation: str,
    output: Path | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    for name in THREAD_CAPS:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"launcher requires {name}=1")
    runner_source = _read_fixed_runner_source()
    request: dict[str, object] = {
        "operation": operation,
        "module_name": _SEALED_MODULE_NAME,
        "runner_path": str(CANONICAL_ROOT / RUNNER_RELATIVE_PATH),
        "runner_source_b64": base64.b64encode(runner_source).decode("ascii"),
        "runner_source_sha256": EXPECTED_RUNNER_SOURCE_SHA256,
    }
    if output is not None:
        request["output"] = str(output)
    if workers is not None:
        request["workers"] = workers
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-s", "-c", _CHILD_BOOTSTRAP],
        input=json.dumps(request, sort_keys=True, separators=(",", ":")),
        cwd=CANONICAL_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "sealed G3 runner failed:\n" + completed.stderr.strip()
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("sealed G3 runner returned non-JSON output") from exc
    if not isinstance(result, dict):
        raise RuntimeError("sealed G3 runner returned a non-object result")
    return result


def sealed_bootstrap_probe() -> dict[str, Any]:
    """CPU-only proof that the frozen runner bytes execute in the child."""

    return _invoke_sealed_child(operation="probe")


def run(*, output: Path, workers: int) -> dict[str, Any]:
    if isinstance(workers, bool) or not 1 <= workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    output = output.resolve()
    if not output.is_relative_to(CANONICAL_ROOT) or output.is_symlink():
        raise ValueError("output must be a non-symlink path inside the repository")
    if output.exists():
        raise FileExistsError(f"exact-equivalence output already exists: {output}")
    return _invoke_sealed_child(operation="run", output=output, workers=workers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    result = run(output=args.output, workers=args.workers)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "content_sha256": result["content_sha256"],
                "scene_count": result["scene_count"],
                "claim_endpoints_retained": result["claim_endpoints_retained"],
                "beacon_count": result["beacon_count"],
                "candidate_conservative_equivalence_pass": result[
                    "candidate_conservative_equivalence_pass"
                ],
                "legacy_strict_binary_equivalence_pass": result[
                    "legacy_strict_binary_equivalence_pass"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
