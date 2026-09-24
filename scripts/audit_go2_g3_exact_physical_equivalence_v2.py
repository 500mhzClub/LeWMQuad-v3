#!/usr/bin/env python3
"""One-shot isolated CLI for the captured G3 V2 audit candidate.

Importing this file intentionally exposes no loader, callback, runtime, path,
hash, probe, or audit helper. Execution authority exists only in the fresh CLI
process below, where every path and digest is lexical and fixed.
"""

if __name__ == "__main__":
    import argparse
    import base64
    import hashlib
    import json
    import os
    from pathlib import Path
    import stat
    import subprocess
    import sys

    canonical_root = Path("/home/andrewknowles/Workspace/LeWMQuad-v3").resolve()
    expected_launcher = (
        canonical_root / "scripts/audit_go2_g3_exact_physical_equivalence_v2.py"
    )
    runner_path = (
        canonical_root
        / "lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v2.py"
    )
    runner_sha256 = "d759cb7fa395646d435bdd0af220a098d7d1e908970a30c4f17fc9e391c296e8"
    output = (
        canonical_root
        / ".generated/go2_g3_exact_physical_equivalence/v2/candidate.json"
    )
    v1_output = (
        canonical_root
        / ".generated/go2_g3_exact_physical_equivalence/v1/candidate.json"
    )
    sealed_module_name = "_lewm_captured_g3_exact_physical_runner_v2"
    thread_caps = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="run only the source/captured-worker bootstrap proof",
    )
    arguments = parser.parse_args()
    if Path(__file__).resolve() != expected_launcher:
        raise PermissionError("G3 V2 launcher is not at its canonical fixed path")
    if isinstance(arguments.workers, bool) or not 1 <= arguments.workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    if output == v1_output:
        raise AssertionError("G3 V2 output aliases the immutable V1 output")
    if not arguments.probe and output.exists():
        raise FileExistsError(f"G3 V2 output already exists: {output}")
    for variable in thread_caps:
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"G3 V2 launcher requires {variable}=1")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(runner_path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PermissionError("captured G3 V2 runner is not singly linked")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    runner_source = b"".join(chunks)
    if hashlib.sha256(runner_source).hexdigest() != runner_sha256:
        raise PermissionError("captured G3 V2 runner hash is not frozen")

    request = {
        "operation": "probe" if arguments.probe else "run",
        "module_name": sealed_module_name,
        "runner_path": str(runner_path),
        "runner_source_b64": base64.b64encode(runner_source).decode("ascii"),
        "runner_source_sha256": runner_sha256,
        "output": str(output),
        "workers": arguments.workers,
    }
    child_bootstrap = r"""
import base64, hashlib, json, pathlib, sys, types
request = json.loads(sys.stdin.read())
payload = base64.b64decode(request["runner_source_b64"], validate=True)
expected = request["runner_source_sha256"]
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError("captured G3 V2 runner bytes changed in transit")
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
    raise RuntimeError("unsupported sealed G3 V2 operation")
print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-s", "-c", child_bootstrap],
        input=json.dumps(request, sort_keys=True, separators=(",", ":")),
        cwd=canonical_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("sealed G3 V2 runner failed:\n" + completed.stderr.strip())
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise RuntimeError("sealed G3 V2 runner returned a non-object")
    if arguments.probe:
        print(
            json.dumps(result, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "content_sha256": result["content_sha256"],
                    "scene_count": result["scene_count"],
                    "claim_endpoints_retained": result[
                        "claim_endpoints_retained"
                    ],
                    "beacon_count": result["beacon_count"],
                    "candidate_v2_exact_equivalence_pass": result[
                        "candidate_v2_exact_equivalence_pass"
                    ],
                    "production_promotion_authorized": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
