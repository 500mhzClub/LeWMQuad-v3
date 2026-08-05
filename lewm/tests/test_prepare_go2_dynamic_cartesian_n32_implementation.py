from __future__ import annotations

from pathlib import Path

import pytest

from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract
from scripts import prepare_go2_dynamic_cartesian_n32_implementation as preparer


def test_manifest_payload_satisfies_the_pure_contract() -> None:
    entries = [
        {"role": role, "path": path, "sha256": format(index + 1, "064x")}
        for index, (role, path) in enumerate(
            sorted(contract.IMPLEMENTATION_SOURCE_PATHS.items())
        )
    ]
    tests = {
        "command": contract.IMPLEMENTATION_TEST_COMMAND,
        "passed": contract.IMPLEMENTATION_TEST_PASSED,
        "all_passed": True,
    }
    manifest = preparer._manifest_payload(
        source_entries=entries,
        tests=tests,
        initial_state_sha256={str(seed): "a" * 64 for seed in contract.EXPECTED_SEEDS},
        state_contract_sha256={str(seed): "b" * 64 for seed in contract.EXPECTED_SEEDS},
    )
    assert contract.validate_implementation_manifest(manifest) == manifest


def test_frozen_pytest_count_parser_is_unambiguous() -> None:
    assert preparer._frozen_test_count("79 passed in 1.23s\n") == 79
    with pytest.raises(RuntimeError, match="unambiguous"):
        preparer._frozen_test_count("no tests ran\n")
    with pytest.raises(RuntimeError, match="unambiguous"):
        preparer._frozen_test_count("1 passed in 1s\n2 passed in 2s\n")


def test_parser_requires_the_canonical_fresh_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = (tmp_path / "implementation.json").resolve()
    monkeypatch.setattr(preparer, "CANONICAL_OUTPUT", output)
    parsed = preparer._parse_args(("--output", str(output)))
    assert parsed.output == output and parsed.device == "cuda:0"
    output.write_text("{}\n")
    with pytest.raises(SystemExit):
        preparer._parse_args(("--output", str(output)))
