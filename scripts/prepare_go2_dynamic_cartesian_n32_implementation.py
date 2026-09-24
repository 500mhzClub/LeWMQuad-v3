#!/usr/bin/env python3
"""Publish the frozen N32 implementation manifest and GPU0 state commitments."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract  # noqa: E402
from scripts import run_go2_dynamic_cartesian_n32 as runner  # noqa: E402


CANONICAL_OUTPUT = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_dynamic_cartesian_n32_v1_implementation_manifest_2026-07-11.json"
).resolve()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        args.output = runner._canonical_path(args.output, name="manifest output path")
    except ValueError as exc:
        parser.error(str(exc))
    if args.output != CANONICAL_OUTPUT:
        parser.error("implementation manifest output path is not canonical")
    if args.output.exists():
        parser.error("implementation manifest already exists and is immutable")
    return args


def _frozen_test_count(output: str) -> int:
    matches = re.findall(r"(?m)(\d+) passed(?:,| in)", output)
    if len(matches) != 1:
        raise RuntimeError("frozen pytest output lacks one unambiguous pass count")
    return int(matches[0])


def _run_frozen_tests() -> dict[str, Any]:
    completed = subprocess.run(
        shlex.split(contract.IMPLEMENTATION_TEST_COMMAND),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"frozen implementation tests failed:\n{output[-4000:]}")
    passed = _frozen_test_count(output)
    if passed != contract.IMPLEMENTATION_TEST_PASSED:
        raise RuntimeError(
            f"frozen test count changed: {passed} != {contract.IMPLEMENTATION_TEST_PASSED}"
        )
    return {
        "command": contract.IMPLEMENTATION_TEST_COMMAND,
        "passed": passed,
        "all_passed": True,
    }


def _source_entries() -> list[dict[str, str]]:
    paths = runner._source_path_contract()
    if set(paths) != set(runner.SOURCE_ROLES) or set(paths) != set(
        contract.IMPLEMENTATION_SOURCE_PATHS
    ):
        raise RuntimeError("implementation source roles changed")
    with ThreadPoolExecutor(max_workers=runner.SOURCE_WORKERS) as pool:
        hashes = dict(
            zip(
                runner.SOURCE_ROLES,
                pool.map(lambda role: runner._sha256_file(paths[role]), runner.SOURCE_ROLES),
                strict=True,
            )
        )
    return [
        {"role": role, "path": str(paths[role]), "sha256": hashes[role]}
        for role in sorted(paths)
    ]


def _input_hashes() -> dict[str, str]:
    inputs = runner._input_contract()
    names = tuple(sorted(inputs))
    with ThreadPoolExecutor(max_workers=runner.SOURCE_WORKERS) as pool:
        observed = dict(
            zip(
                names,
                pool.map(
                    lambda name: runner._sha256_file(Path(inputs[name]["path"])),
                    names,
                ),
                strict=True,
            )
        )
    expected = {name: str(inputs[name]["sha256"]) for name in names}
    if observed != expected:
        raise RuntimeError("bound N32 input hash changed")
    return observed


def _manifest_payload(
    *,
    source_entries: Sequence[Mapping[str, str]],
    tests: Mapping[str, Any],
    initial_state_sha256: Mapping[str, str],
    state_contract_sha256: Mapping[str, str],
) -> dict[str, Any]:
    entries = [dict(entry) for entry in source_entries]
    core = {
        "schema": contract.IMPLEMENTATION_MANIFEST_SCHEMA,
        "binding": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS["binding"],
            "sha256": contract.EXECUTION_BINDING_SHA256,
        },
        "preoutput_amendment": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS["preoutput_amendment"],
            "sha256": contract.PREOUTPUT_AMENDMENT_SHA256,
        },
        "attempt_control_amendment": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS[
                "attempt_control_amendment"
            ],
            "sha256": contract.ATTEMPT_CONTROL_AMENDMENT_SHA256,
        },
        "sources": {
            "entries": entries,
            "entry_count": len(entries),
            "source_map_sha256": contract.canonical_json_sha256(entries),
        },
        "tests": dict(tests),
        "inputs": contract.INPUT_BINDINGS,
        "resource_policy": contract.RESOURCE_POLICY,
        "model_config": contract.MODEL_CONFIG,
        "objective": contract.OBJECTIVE_CONTRACT,
        "preprocessing": contract.PREPROCESSING_CONTRACT,
        "controls": contract.CONTROL_CONTRACT,
        "projective_query_support": contract.PROJECTIVE_QUERY_SUPPORT,
        "model_initial_state_sha256": dict(initial_state_sha256),
        "model_state_contract_sha256": dict(state_contract_sha256),
        "schedules": contract.SCHEDULE_CONTRACT,
        "commands": contract.COMMAND_CONTRACT,
    }
    return {**core, "content_sha256": contract.canonical_json_sha256(core)}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sources_pretest = _source_entries()
    inputs_pretest = _input_hashes()
    tests = _run_frozen_tests()
    sources_before = _source_entries()
    inputs_before = _input_hashes()
    if sources_before != sources_pretest or inputs_before != inputs_pretest:
        raise RuntimeError("N32 sources or inputs changed while tests ran")
    device, _device_record = runner._validate_resource_environment(str(args.device))

    state_hashes: dict[str, str] = {}
    state_contract_hashes: dict[str, str] = {}
    for seed in contract.EXPECTED_SEEDS:
        runner._configure_determinism(seed)
        _state, state_sha256, proof = runner._derive_initial_state(device, seed)
        state_hashes[str(seed)] = state_sha256
        state_contract_hashes[str(seed)] = proof["state_contract_sha256"]
        del _state

    sources_after = _source_entries()
    inputs_after = _input_hashes()
    if sources_after != sources_before or inputs_after != inputs_before:
        raise RuntimeError("N32 sources or inputs changed during manifest preparation")
    manifest = _manifest_payload(
        source_entries=sources_after,
        tests=tests,
        initial_state_sha256=state_hashes,
        state_contract_sha256=state_contract_hashes,
    )
    contract.validate_implementation_manifest(manifest)
    expected_file_sha256 = hashlib.sha256(
        runner._published_json_bytes(manifest)
    ).hexdigest()
    runner._publish_json_exclusive(args.output, manifest)
    observed = runner._read_json(
        args.output,
        expected_sha256=expected_file_sha256,
        name="N32 implementation manifest",
    )
    contract.validate_implementation_manifest(observed)
    if _source_entries() != sources_after or _input_hashes() != inputs_after:
        raise RuntimeError("N32 sources or inputs changed before manifest publication")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": expected_file_sha256,
                "content_sha256": manifest["content_sha256"],
                "seeds": list(contract.EXPECTED_SEEDS),
                "model_forward_calls": 0,
                "model_output_frames": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
