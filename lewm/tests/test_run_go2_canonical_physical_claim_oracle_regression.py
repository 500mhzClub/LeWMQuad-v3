from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/run_go2_canonical_physical_claim_oracle_regression.py"
SPEC = importlib.util.spec_from_file_location("canonical_claim_oracle_runner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_map() -> dict:
    entries = [
        {
            "role": role,
            "path": path,
            "sha256": _sha256(ROOT / path),
        }
        for role, path in sorted(runner.SOURCE_PATHS.items())
    ]
    return {
        "entries": entries,
        "source_map_sha256": runner._canonical_sha256(entries),
    }


def _input_bindings() -> dict:
    hashes = {
        "development_manifest": runner.EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        "materialization": runner.EXPECTED_MATERIALIZATION_SHA256,
        "geometry": runner.EXPECTED_GEOMETRY_FILE_SHA256,
        "primitive_registry": runner.EXPECTED_PRIMITIVE_REGISTRY_SHA256,
        "directional_policy": runner.EXPECTED_DIRECTIONAL_POLICY_FILE_SHA256,
        "prior_comparator": runner.EXPECTED_PRIOR_COMPARATOR_SHA256,
    }
    result = {}
    for role, path in runner.INPUT_PATHS.items():
        record = {
            "path": path,
            "sha256": hashes[role],
            "access_mode": (
                "identity_only_do_not_open"
                if role == "prior_comparator"
                else "hash_then_parse"
                if role
                in {
                    "development_manifest",
                    "geometry",
                    "primitive_registry",
                    "directional_policy",
                }
                else "hash_only"
            ),
        }
        if role == "directional_policy":
            record["content_sha256"] = (
                runner.EXPECTED_DIRECTIONAL_POLICY_CONTENT_SHA256
            )
        result[role] = record
    return result


def _implementation_payload() -> dict:
    payload = {
        "schema": runner.IMPLEMENTATION_MANIFEST_SCHEMA,
        "binding_sha256": runner.EXPECTED_BINDING_SHA256,
        "source_map": _source_map(),
        "input_bindings": _input_bindings(),
        "command_identity": deepcopy(runner.COMMAND_IDENTITY_TEMPLATE),
        "oracle_config": {"synthetic_config_binding": True},
        "eligibility_config": {"synthetic_eligibility_binding": True},
        "exclusive_output": {
            "path": str(runner.OUTPUT_PATH.relative_to(ROOT)),
            "schema": runner.RESULT_SCHEMA,
            "atomic_no_replace": True,
        },
    }
    payload["content_sha256"] = runner._content_sha256(payload)
    return payload


def _rehash_manifest(payload: dict) -> dict:
    payload["content_sha256"] = runner._content_sha256(payload)
    return payload


def _rehash_source_map(payload: dict) -> dict:
    payload["source_map"]["source_map_sha256"] = runner._canonical_sha256(
        payload["source_map"]["entries"]
    )
    return _rehash_manifest(payload)


def test_bootstrap_has_no_project_import_before_source_verification() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(("lewm", "lewm_worlds", "lewm_genesis")):
                forbidden.append(node.module)
        if isinstance(node, ast.Import):
            forbidden.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith(("lewm", "lewm_worlds", "lewm_genesis"))
            )
    assert forbidden == []
    source = ast.get_source_segment(
        SCRIPT_PATH.read_text(encoding="utf-8"),
        next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_authoritative_regression"
        ),
    )
    assert source is not None
    assert source.index("load_and_verify_implementation_manifest") < source.index(
        "_load_project_api()"
    )


def test_authoritative_worker_contract_is_fixed_and_wired_to_both_scene_stages() -> None:
    assert runner.COMMAND_IDENTITY_TEMPLATE["worker_pool"] == {
        "kind": "spawn_process",
        "worker_count": 6,
        "threads_per_worker": 1,
        "merge_order": "development_manifest_index",
        "worker_runtime_input_file_access": False,
    }
    source = inspect.getsource(runner.run_authoritative_regression)
    assert source.count("workers=WORKER_COUNT") == 2
    assert "preloaded_scene_manifests=scene_manifests" in source
    assert source.count("api.policy_from_geometry_contract(") == 1
    assert "preloaded_directional_policy=policy" in source
    assert source.index("_load_project_api()") < source.index(
        "run_parallel_eligibility_jobs("
    )


def test_authoritative_runner_loads_policy_once_and_reuses_exact_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    @dataclass(frozen=True)
    class FakeOracleConfig:
        synthetic_config_binding: bool = True

    @dataclass(frozen=True)
    class FakeEligibilityConfig:
        synthetic_eligibility_binding: bool = True

    oracle_config = FakeOracleConfig()
    eligibility_config = FakeEligibilityConfig()
    implementation = SimpleNamespace(
        sha256="2" * 64,
        payload={
            "oracle_config": asdict(oracle_config),
            "eligibility_config": asdict(eligibility_config),
        },
        source_map={},
        input_bindings={},
    )
    geometry = SimpleNamespace(schema="synthetic_geometry")
    policy = SimpleNamespace(
        content_sha256=runner.EXPECTED_DIRECTIONAL_POLICY_CONTENT_SHA256
    )
    calls = {"policy_loads": 0}
    observed: dict[str, object] = {}

    class OracleConfigFactory:
        @staticmethod
        def from_geometry_contract(_geometry):
            return oracle_config

    class RegistryFactory:
        @staticmethod
        def from_yaml(_path):
            return object()

    def load_policy(_geometry, *, repository_root):
        assert repository_root == runner.ROOT
        calls["policy_loads"] += 1
        return policy

    def run_suite(**kwargs):
        observed["oracle"] = kwargs["preloaded_directional_policy"]
        return {"schema": "synthetic_oracle"}

    def run_eligibility(**kwargs):
        observed["eligibility"] = kwargs["policy"]
        return {"scene": {"scene_id": "scene"}}

    finalized_payload = {"schema": "synthetic_finalized"}
    api = SimpleNamespace(
        OracleConfig=OracleConfigFactory,
        PrimitiveRegistry=RegistryFactory,
        finalize_suite=lambda *_args, **_kwargs: SimpleNamespace(
            passed=True,
            errors=(),
            finalized_payload=finalized_payload,
        ),
        load_geometry_contract=lambda *_args, **_kwargs: geometry,
        physical_config_from_geometry_contract=lambda _geometry: eligibility_config,
        policy_from_geometry_contract=load_policy,
        run_development_suite=run_suite,
    )

    monkeypatch.setattr(runner, "OUTPUT_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(
        runner,
        "load_and_verify_implementation_manifest",
        lambda _sha: implementation,
    )
    monkeypatch.setattr(
        runner,
        "verify_runtime_input_files",
        lambda _implementation: {},
    )
    monkeypatch.setattr(runner, "_load_project_api", lambda: api)
    monkeypatch.setattr(
        runner,
        "_load_development_protocol",
        lambda: {
            "validation_scenes": [
                {
                    "scene_id": "scene",
                    "family": "family",
                    "manifest_sha256": "3" * 64,
                    "beacon_count": 4,
                }
            ]
        },
    )
    monkeypatch.setattr(
        runner,
        "_load_scene_manifests",
        lambda _api, _records: {"scene": object()},
    )
    monkeypatch.setattr(runner, "run_parallel_eligibility_jobs", run_eligibility)
    monkeypatch.setattr(runner, "_verify_source_map", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_bound_input_identity", lambda _implementation: {})
    monkeypatch.setattr(runner, "_resolved_command_identity", lambda _sha: {})
    monkeypatch.setattr(runner, "publish_finalized_result", lambda result: None)
    for name in runner.CPU_THREAD_CAP_ENV:
        monkeypatch.setenv(name, "1")

    result = runner.run_authoritative_regression(implementation.sha256)
    assert result is finalized_payload
    assert calls["policy_loads"] == 1
    assert observed["oracle"] is policy
    assert observed["eligibility"] is policy


def test_worker_merge_is_schedule_independent_and_identity_checked() -> None:
    scene_ids = ["scene_z", "scene_a", "scene_m"]
    shuffled = [
        (2, {"scene_id": "scene_m", "value": 2}),
        (0, {"scene_id": "scene_z", "value": 0}),
        (1, {"scene_id": "scene_a", "value": 1}),
    ]
    assert runner.merge_indexed_payloads(
        shuffled, expected_scene_ids=scene_ids
    ) == [
        {"scene_id": "scene_z", "value": 0},
        {"scene_id": "scene_a", "value": 1},
        {"scene_id": "scene_m", "value": 2},
    ]
    with pytest.raises(runner.ReadinessError, match="duplicated"):
        runner.merge_indexed_payloads(
            [shuffled[0], shuffled[0], shuffled[1]],
            expected_scene_ids=scene_ids,
        )
    with pytest.raises(runner.ReadinessError, match="scene identity"):
        runner.merge_indexed_payloads(
            [
                (0, {"scene_id": "scene_a"}),
                (1, {"scene_id": "scene_a"}),
                (2, {"scene_id": "scene_m"}),
            ],
            expected_scene_ids=scene_ids,
        )
    with pytest.raises(runner.ReadinessError, match="outside"):
        runner.merge_indexed_payloads(
            [
                (0, {"scene_id": "scene_z"}),
                (1, {"scene_id": "scene_a"}),
                (3, {"scene_id": "other"}),
            ],
            expected_scene_ids=scene_ids,
        )
    with pytest.raises(runner.ReadinessError, match="incomplete"):
        runner.merge_indexed_payloads(shuffled[:2], expected_scene_ids=scene_ids)


def test_parent_thread_caps_are_exact_and_restored(monkeypatch) -> None:
    original = {}
    for index, name in enumerate(runner.CPU_THREAD_CAP_ENV):
        value = None if index % 2 else str(index + 2)
        original[name] = value
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    previous = runner.configure_single_thread_cpu_worker()
    assert previous == original
    assert all(os.environ[name] == "1" for name in runner.CPU_THREAD_CAP_ENV)
    runner.restore_cpu_thread_environment(previous)
    assert {name: os.environ.get(name) for name in runner.CPU_THREAD_CAP_ENV} == original


def test_complete_live_source_map_and_implementation_payload_validate() -> None:
    payload = _implementation_payload()
    verified = runner._verify_implementation_payload(
        payload,
        manifest_sha256="2" * 64,
    )
    assert verified.source_map == payload["source_map"]
    assert verified.input_bindings == payload["input_bindings"]


@pytest.mark.parametrize("mutation", ("missing", "extra", "path", "hash", "order"))
def test_source_map_mutations_fail_closed(mutation: str) -> None:
    payload = _implementation_payload()
    entries = payload["source_map"]["entries"]
    if mutation == "missing":
        entries.pop()
    elif mutation == "extra":
        entries.append(
            {"role": "extra", "path": "extra.py", "sha256": "0" * 64}
        )
    elif mutation == "path":
        entries[0]["path"] = entries[1]["path"]
    elif mutation == "hash":
        entries[0]["sha256"] = "0" * 64
    else:
        entries[0], entries[1] = entries[1], entries[0]
    _rehash_source_map(payload)
    with pytest.raises(runner.ReadinessError):
        runner._verify_implementation_payload(payload, manifest_sha256="2" * 64)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["input_bindings"].pop("geometry"),
        lambda value: value["input_bindings"]["geometry"].update(sha256="0" * 64),
        lambda value: value["input_bindings"]["primitive_registry"].update(
            sha256="0" * 64
        ),
        lambda value: value["input_bindings"]["prior_comparator"].update(
            access_mode="hash_only"
        ),
        lambda value: value["command_identity"].update(executable="python3"),
        lambda value: value["command_identity"]["worker_pool"].update(
            worker_runtime_input_file_access=0
        ),
        lambda value: value["exclusive_output"].update(atomic_no_replace=False),
        lambda value: value["exclusive_output"].update(atomic_no_replace=1),
        lambda value: value.update(binding_sha256="0" * 64),
    ),
)
def test_binding_command_input_and_output_mutations_fail_closed(mutate) -> None:
    payload = _implementation_payload()
    mutate(payload)
    _rehash_manifest(payload)
    with pytest.raises(runner.ReadinessError):
        runner._verify_implementation_payload(payload, manifest_sha256="2" * 64)


def test_manifest_file_hash_and_duplicate_json_keys_fail_before_source_use(
    tmp_path: Path,
) -> None:
    path = tmp_path / "implementation.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    sha = _sha256(path)
    with pytest.raises(runner.ReadinessError, match="duplicate"):
        runner.load_and_verify_implementation_manifest(sha, manifest_path=path)
    with pytest.raises(runner.ReadinessError, match="file SHA-256"):
        runner.load_and_verify_implementation_manifest("0" * 64, manifest_path=path)


def test_prior_comparator_is_never_hashed_or_opened(tmp_path: Path) -> None:
    implementation = runner.VerifiedImplementation(
        sha256="2" * 64,
        payload={},
        source_map={},
        input_bindings=_input_bindings(),
        command_identity={},
    )
    expected_by_path = {}
    for role, record in implementation.input_bindings.items():
        if role == "prior_comparator":
            continue
        path = tmp_path / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode("ascii"))
        expected_by_path[path] = record["sha256"]
    opened = []

    def fake_hash(path: Path) -> str:
        opened.append(path)
        return expected_by_path[path]

    ledger = runner.verify_runtime_input_files(
        implementation, root=tmp_path, file_hasher=fake_hash
    )
    assert ledger["prior_comparator_hash_reads"] == 0
    assert tmp_path / runner.INPUT_PATHS["prior_comparator"] not in opened


def test_source_failure_prevents_input_verification_and_project_import(monkeypatch) -> None:
    calls = []

    def fail_source(_sha):
        calls.append("source")
        raise runner.ReadinessError("source failure")

    monkeypatch.setattr(runner, "load_and_verify_implementation_manifest", fail_source)
    monkeypatch.setattr(
        runner,
        "verify_runtime_input_files",
        lambda _implementation: calls.append("inputs"),
    )
    monkeypatch.setattr(runner, "_load_project_api", lambda: calls.append("imports"))
    with pytest.raises(runner.ReadinessError, match="source failure"):
        runner.run_authoritative_regression("2" * 64)
    assert calls == ["source"]


def test_input_failure_prevents_project_import(monkeypatch, tmp_path: Path) -> None:
    calls = []
    implementation = runner.VerifiedImplementation(
        sha256="2" * 64,
        payload={},
        source_map={},
        input_bindings={},
        command_identity={},
    )
    monkeypatch.setattr(runner, "OUTPUT_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(
        runner,
        "load_and_verify_implementation_manifest",
        lambda _sha: calls.append("source") or implementation,
    )

    def fail_inputs(_implementation):
        calls.append("inputs")
        raise runner.ReadinessError("input failure")

    monkeypatch.setattr(runner, "verify_runtime_input_files", fail_inputs)
    monkeypatch.setattr(runner, "_load_project_api", lambda: calls.append("imports"))
    with pytest.raises(runner.ReadinessError, match="input failure"):
        runner.run_authoritative_regression("2" * 64)
    assert calls == ["source", "inputs"]


def test_atomic_no_replace_preserves_existing_bytes(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    runner.publish_json_atomic_no_replace(output, {"schema": "first"})
    first = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        runner.publish_json_atomic_no_replace(output, {"schema": "second"})
    assert output.read_bytes() == first


def test_failed_finalization_creates_no_output(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    failed = SimpleNamespace(
        passed=False,
        errors=("synthetic_failure",),
        finalized_payload=None,
    )
    with pytest.raises(runner.ReadinessError, match="synthetic_failure"):
        runner.publish_finalized_result(failed, path=output)
    assert not output.exists()


def test_cli_exposes_only_required_manifest_hash() -> None:
    parsed = runner._parse_args(["--implementation-manifest-sha256", "2" * 64])
    assert vars(parsed) == {"implementation_manifest_sha256": "2" * 64}
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--implementation-manifest-sha256",
                "2" * 64,
                "--output",
                "other.json",
            ]
        )
    with pytest.raises(SystemExit):
        runner._parse_args([f"--implementation-manifest-sha256={'2' * 64}"])
