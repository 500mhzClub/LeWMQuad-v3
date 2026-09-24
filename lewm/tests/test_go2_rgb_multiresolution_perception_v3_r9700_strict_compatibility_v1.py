from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace
from typing import Iterator

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
LAUNCHER_PATH = (
    ROOT
    / "scripts/launch_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_multiresolution_perception_v3_"
    "r9700_strict_compatibility_v1.py"
)
TEST_PATH = Path(__file__).resolve()


@contextmanager
def _forbid_torch_import() -> Iterator[None]:
    original = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("source-only tests must not import Torch")
        return original(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        yield
    finally:
        builtins.__import__ = original


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with _forbid_torch_import():
    contract = _load(CONTRACT_PATH, "_test_v3_r9700_contract")
    launcher = _load(LAUNCHER_PATH, "_test_v3_r9700_launcher")
    runner = _load(RUNNER_PATH, "_test_v3_r9700_runner")


def _digest(character: str) -> str:
    return character * 64


def _binding(path: str, character: str, *, content: bool = True) -> dict[str, object]:
    value: dict[str, object] = {
        "path": path,
        "file_sha256": _digest(character),
        "byte_count": 100,
    }
    if content:
        value["content_sha256"] = _digest(chr(ord(character) + 1))
    return value


def _sources() -> dict[str, str]:
    return {
        path: hashlib.sha256(path.encode("ascii")).hexdigest()
        for path in contract.SOURCE_PATHS
    }


def _review(
    sources: dict[str, str] | None = None,
    *,
    reviewer: str = "/root/independent_checker_reviewer",
) -> dict[str, object]:
    bindings = _sources() if sources is None else sources
    core = {
        "schema": contract.REVIEW_SCHEMA,
        "status": contract.REVIEW_STATUS,
        "reviewer": reviewer,
        "reviewed_source_commit": "1" * 40,
        "source_paths": list(contract.SOURCE_PATHS),
        "source_bindings": bindings,
        "source_bindings_sha256": contract.source_bindings_sha256(bindings),
        "preregistration": dict(contract.PREREGISTRATION_BINDING),
        "decision": dict(contract.DECISION_BINDING),
        "declared_candidate_source_witnesses":
            dict(contract.DECLARED_CANDIDATE_SOURCE_WITNESSES),
        "prior_strict_failure_audit_witness":
            dict(contract.PRIOR_STRICT_FAILURE_AUDIT_WITNESS),
        "operation_contract_sha256": contract.OPERATION_CONTRACT_SHA256,
        "output_contract_sha256": contract.OUTPUT_CONTRACT_SHA256,
        "findings": [],
        "authority": dict(contract.REVIEW_AUTHORITY),
    }
    return contract.with_content_sha256(core)


def _authorization(
    review_binding: dict[str, object],
    sources: dict[str, str] | None = None,
    *,
    reviewer: str = "/root/independent_checker_reviewer",
    authorizer: str = "/root/independent_checker_authorizer",
) -> dict[str, object]:
    bindings = _sources() if sources is None else sources
    core = {
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": contract.AUTHORIZATION_STATUS,
        "authorizer": authorizer,
        "reviewer": reviewer,
        "source_review": review_binding,
        "source_bindings_sha256": contract.source_bindings_sha256(bindings),
        "operation_contract_sha256": contract.OPERATION_CONTRACT_SHA256,
        "output_contract_sha256": contract.OUTPUT_CONTRACT_SHA256,
        "attempt_index": 1,
        "maximum_attempts": 1,
        "output_root": contract.OUTPUT_ROOT_RELATIVE_PATH,
        "authority": dict(contract.EXECUTION_AUTHORITY),
    }
    return contract.with_content_sha256(core)


def _source_authority() -> dict[str, object]:
    return {
        "source_binding_count": 4,
        "source_bindings_sha256": contract.source_bindings_sha256(_sources()),
        "preregistration": dict(contract.PREREGISTRATION_BINDING),
        "decision": dict(contract.DECISION_BINDING),
        "source_review": _binding(contract.REVIEW_RELATIVE_PATH, "2"),
        "execution_authorization":
            _binding(contract.AUTHORIZATION_RELATIVE_PATH, "4"),
        "generated_runtime_input_open_count": 0,
        "model_or_runtime_root_open_count": 0,
        "torch_imported": False,
    }


def _python_identity() -> dict[str, object]:
    return {
        "implementation": "cpython",
        "version": "3.12.0 test",
        "cache_tag": "cpython-312",
        "executable": "/usr/bin/python3",
        "isolated": True,
        "dont_write_bytecode": True,
    }


def _stack_identity() -> dict[str, object]:
    return {
        "torch_version": "2.10.0.dev+rocm6.3",
        "torch_git_version": "abc",
        "hip_version": "6.3",
    }


def _device_identity() -> dict[str, object]:
    return {
        "visible_device_count": 1,
        "visible_device_index": 0,
        "visible_device_name": "AMD Radeon AI PRO R9700",
        "total_memory_bytes": 34_208_743_424,
    }


def _preflight(
    source_authority: dict[str, object] | None = None,
    *,
    launcher_process_id: int | None = None,
) -> dict[str, object]:
    core = {
        "schema": contract.PREFLIGHT_SCHEMA,
        "status": contract.PREFLIGHT_STATUS,
        "launcher_process_id":
            os.getpid() if launcher_process_id is None else launcher_process_id,
        "preflight_child_process_id": 12345,
        "python": _python_identity(),
        "stack": _stack_identity(),
        "device": _device_identity(),
        "tensor_allocation_count": 0,
        "memory_allocated_bytes": 0,
        "memory_reserved_bytes": 0,
        "payload_open_count": 0,
        "model_or_runtime_root_open_count": 0,
        "source_authority":
            _source_authority() if source_authority is None else source_authority,
        "launcher_source_sha256": _sources()[contract.LAUNCHER_RELATIVE_PATH],
        "immediate_exec_required": True,
        "intervening_gpu_query_count": 0,
    }
    return contract.with_content_sha256(core)


def _subprobe(
    operation: str,
    *,
    outcome: str = "PASS",
    message: str | None = None,
    stage: str | None = None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    spec = (
        contract.GRID_OPERATION
        if operation == "grid_sample"
        else contract.SCATTER_OPERATION
    )
    if outcome == "PASS":
        exception = None
        status = "PASS"
    else:
        if message is None:
            message = (
                contract.EXPECTED_GRID_STRICT_ERROR
                if operation == "grid_sample"
                else contract.EXPECTED_SCATTER_STRICT_ERROR
            )
        exception = {
            "type": "RuntimeError",
            "message": message,
            "message_sha256": hashlib.sha256(
                message.encode("utf-8")
            ).hexdigest(),
        }
        status = "EXCEPTION"
    checks, counts, exact_stage = contract._expected_subprobe_state(
        operation,
        "PASS" if outcome == "PASS" else "COMPATIBILITY_FAIL",
    )
    receipt_stage = exact_stage if stage is None else stage
    core = {
        "schema": contract.SUBPROBE_SCHEMA,
        "operation": operation,
        "execution_order": spec["execution_order"],
        "status": status,
        "stage": receipt_stage,
        "python": _python_identity(),
        "stack": _stack_identity(),
        "device": _device_identity(),
        "determinism": dict(contract.DETERMINISM_CONTRACT),
        "operation_contract_sha256": contract.OPERATION_CONTRACT_SHA256,
        "operation_spec": spec,
        "warnings": [] if warnings is None else warnings,
        "exception": exception,
        "checks": checks,
        "counts": counts,
    }
    return contract.with_content_sha256(core)


def _canonical_raw(value: dict[str, object]) -> bytes:
    return contract.canonical_json_bytes(value) + b"\n"


def _make_warning(message: str = "unexpected") -> dict[str, str]:
    return {
        "category": "UserWarning",
        "message": message,
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
    }


def _unseal(path: Path) -> None:
    if not path.exists():
        return
    os.chmod(path, 0o700)
    for child in path.iterdir():
        if child.is_file() and not child.is_symlink():
            os.chmod(child, 0o600)


def test_contract_is_pure_stdlib_and_source_closure_is_exact() -> None:
    assert "torch" not in contract.__dict__
    assert tuple(contract.SOURCE_PATHS) == (
        contract.CONTRACT_RELATIVE_PATH,
        contract.LAUNCHER_RELATIVE_PATH,
        contract.RUNNER_RELATIVE_PATH,
        contract.TEST_RELATIVE_PATH,
    )
    assert set(contract.current_source_bindings(ROOT)) == set(contract.SOURCE_PATHS)
    assert contract.OUTPUT_ROOT_RELATIVE_PATH == (
        ".generated/"
        "go2_rgb_multiresolution_perception_r9700_strict_compatibility_v1"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != contract.V3_PROBE_ROOT_RELATIVE_PATH
    assert not contract.OUTPUT_ROOT_RELATIVE_PATH.startswith(
        contract.V3_PROBE_ROOT_RELATIVE_PATH + "/"
    )
    assert not contract.V3_PROBE_ROOT_RELATIVE_PATH.startswith(
        contract.OUTPUT_ROOT_RELATIVE_PATH + "/"
    )


def test_committed_v3_decision_and_preregistration_bindings_are_exact() -> None:
    decision_raw = (ROOT / contract.DECISION_RELATIVE_PATH).read_bytes()
    assert len(decision_raw) == contract.DECISION_BINDING["byte_count"]
    assert hashlib.sha256(decision_raw).hexdigest() == (
        contract.DECISION_BINDING["file_sha256"]
    )
    preregistration_raw = (
        ROOT / contract.PREREGISTRATION_RELATIVE_PATH
    ).read_bytes()
    assert len(preregistration_raw) == contract.PREREGISTRATION_BINDING["byte_count"]
    assert hashlib.sha256(preregistration_raw).hexdigest() == (
        contract.PREREGISTRATION_BINDING["file_sha256"]
    )
    preregistration = contract.parse_canonical_json(
        preregistration_raw,
        name="committed V3 preregistration",
    )
    assert contract.validate_preregistration(preregistration) == preregistration


@pytest.mark.parametrize("path", [CONTRACT_PATH, LAUNCHER_PATH, RUNNER_PATH])
def test_parent_sources_have_only_stdlib_top_level_imports(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "torch",
        "numpy",
        "PIL",
        "lewm.models",
        "lewm.datasets",
    )
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imported
        for prefix in forbidden_prefixes
    )


def test_only_isolated_children_contain_torch_imports() -> None:
    grid = ast.parse(runner.GRID_CHILD_PROGRAM)
    scatter = ast.parse(runner.SCATTER_CHILD_PROGRAM)
    grid_imports = {
        alias.name
        for node in ast.walk(grid)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(grid)
        if isinstance(node, ast.ImportFrom)
    }
    scatter_imports = {
        alias.name
        for node in ast.walk(scatter)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" in grid_imports
    assert "torch.nn.functional" in grid_imports
    assert "torch" in scatter_imports
    assert "lewm" not in " ".join(sorted(grid_imports | scatter_imports))
    compile(runner.GRID_CHILD_PROGRAM, "<grid-child>", "exec")
    compile(runner.SCATTER_CHILD_PROGRAM, "<scatter-child>", "exec")


def test_operation_contract_matches_exact_real_dispatch_dimensions() -> None:
    grid = contract.GRID_OPERATION
    assert grid["dense_feature_shape"] == [4, 36, 112, 112]
    assert grid["full_query_shape"] == [4, 128, 128, 5, 2]
    assert grid["query_count_per_batch"] == 128 * 128 * 5 == 81920
    assert grid["query_chunk_size"] == 4096
    assert grid["call_count"] == 20
    assert grid["grid_call_shape"] == [4, 4096, 1, 2]
    assert grid["grid_call_output_shape"] == [4, 36, 4096, 1]
    scatter = contract.SCATTER_OPERATION
    assert scatter["ray_count_per_batch"] == 84 * 112 == 9408
    assert 36 * 256 + 192 == 9408
    assert scatter["full_chunk"]["destination_shape"] == [4 * 256 * 4096]
    assert scatter["tail_chunk"]["destination_shape"] == [4 * 192 * 4096]
    assert scatter["full_chunk"]["selected_source_and_index_count"] == 4 * 64 * 256
    assert scatter["tail_chunk"]["selected_source_and_index_count"] == 4 * 64 * 192
    assert scatter["scatter_add_call_count"] == (36 + 1) * 4 == 148


def test_embedded_programs_fix_exact_multiplicity_and_strict_mode() -> None:
    assert "range(0, 81920, 4096)" in runner.GRID_CHILD_PROGRAM
    assert "counts[\"grid_sample_forward_invocation_count\"] == 20" in (
        runner.GRID_CHILD_PROGRAM
    )
    assert "range(36)" in runner.SCATTER_CHILD_PROGRAM
    assert "counts[\"scatter_add_invocation_count\"] == 148" in (
        runner.SCATTER_CHILD_PROGRAM
    )
    for program in (runner.GRID_CHILD_PROGRAM, runner.SCATTER_CHILD_PROGRAM):
        assert "use_deterministic_algorithms(True, warn_only=False)" in program
        assert "warn_only=True" not in program
        assert "use_deterministic_algorithms(False" not in program
        assert "optimizer" not in program.casefold() or (
            '"optimizer_step_count": 0' in program
        )


def test_runner_reserves_before_any_torch_child_and_grid_precedes_scatter() -> None:
    source = inspect.getsource(runner.run_parent)
    reserve = source.index("_reserve_output_root()")
    grid = source.index("program=GRID_CHILD_PROGRAM")
    scatter = source.index("program=SCATTER_CHILD_PROGRAM")
    assert reserve < grid < scatter
    reserve_source = inspect.getsource(runner._reserve_output_root)
    assert "V3_PROBE_ROOT" not in reserve_source
    assert "OUTPUT_ROOT_RELATIVE_PATH" in reserve_source


def test_canonical_json_and_self_hash_reject_mutation() -> None:
    value = contract.with_content_sha256({"schema": "x", "value": [1, 2]})
    raw = _canonical_raw(value)
    assert contract.parse_canonical_json(raw, name="x") == value
    assert contract.validate_self_hash(value) == value
    changed = dict(value)
    changed["value"] = [2, 1]
    with pytest.raises(ValueError, match="self-hash"):
        contract.validate_self_hash(changed)
    with pytest.raises(ValueError, match="canonical"):
        contract.parse_canonical_json(raw.replace(b",", b", ", 1), name="x")


def test_independent_review_and_authorization_validators() -> None:
    sources = _sources()
    review = _review(sources)
    assert contract.validate_review(
        review,
        expected_sources=sources,
        preregistration_binding=contract.PREREGISTRATION_BINDING,
        decision_binding=contract.DECISION_BINDING,
    ) == review
    review_raw = _canonical_raw(review)
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization = _authorization(review_binding, sources)
    assert contract.validate_authorization(
        authorization,
        review_binding=review_binding,
        reviewer=review["reviewer"],
        expected_source_bindings_sha256=contract.source_bindings_sha256(sources),
    ) == authorization


@pytest.mark.parametrize(
    ("reviewer", "authorizer"),
    [
        (contract.IMPLEMENTATION_AUTHOR, "/root/authorizer"),
        ("/root/reviewer", "/root/reviewer"),
        ("/root/reviewer", contract.IMPLEMENTATION_AUTHOR),
    ],
)
def test_review_and_authorization_require_independence(
    reviewer: str,
    authorizer: str,
) -> None:
    sources = _sources()
    review = _review(sources, reviewer=reviewer)
    if reviewer == contract.IMPLEMENTATION_AUTHOR:
        with pytest.raises(PermissionError, match="independent"):
            contract.validate_review(
                review,
                expected_sources=sources,
                preregistration_binding=contract.PREREGISTRATION_BINDING,
                decision_binding=contract.DECISION_BINDING,
            )
        return
    review_raw = _canonical_raw(review)
    binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization = _authorization(
        binding,
        sources,
        reviewer=reviewer,
        authorizer=authorizer,
    )
    with pytest.raises(PermissionError, match="independent"):
        contract.validate_authorization(
            authorization,
            review_binding=binding,
            reviewer=reviewer,
            expected_source_bindings_sha256=
                contract.source_bindings_sha256(sources),
        )


def test_review_rejects_any_source_or_authority_drift() -> None:
    sources = _sources()
    review = _review(sources)
    changed_sources = dict(sources)
    changed_sources[contract.RUNNER_RELATIVE_PATH] = "f" * 64
    with pytest.raises(PermissionError, match="bindings"):
        contract.validate_review(
            review,
            expected_sources=changed_sources,
            preregistration_binding=contract.PREREGISTRATION_BINDING,
            decision_binding=contract.DECISION_BINDING,
        )
    changed = dict(review)
    changed["authority"] = {
        **contract.REVIEW_AUTHORITY,
        "compatibility_run_authorized": True,
    }
    changed = contract.with_content_sha256(
        {key: item for key, item in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(PermissionError, match="bindings"):
        contract.validate_review(
            changed,
            expected_sources=sources,
            preregistration_binding=contract.PREREGISTRATION_BINDING,
            decision_binding=contract.DECISION_BINDING,
        )


def test_preflight_binds_exact_stack_device_and_zero_tensors() -> None:
    authority = _source_authority()
    value = _preflight(authority)
    assert contract.validate_preflight(
        value,
        expected_source_authority=authority,
    ) == value
    for field, bad in (
        ("tensor_allocation_count", 1),
        ("memory_allocated_bytes", 4),
        ("payload_open_count", 1),
        ("intervening_gpu_query_count", 1),
    ):
        core = {
            key: item
            for key, item in value.items()
            if key != "content_sha256"
        }
        core[field] = bad
        with pytest.raises(PermissionError):
            contract.validate_preflight(
                contract.with_content_sha256(core),
                expected_source_authority=authority,
            )


def test_preflight_rejects_wrong_device_or_nonisolated_python() -> None:
    authority = _source_authority()
    value = _preflight(authority)
    for mutate in ("device", "python"):
        core = {
            key: item
            for key, item in value.items()
            if key != "content_sha256"
        }
        if mutate == "device":
            core["device"] = {**_device_identity(), "visible_device_count": 2}
        else:
            core["python"] = {**_python_identity(), "isolated": False}
        with pytest.raises((PermissionError, ValueError)):
            contract.validate_preflight(
                contract.with_content_sha256(core),
                expected_source_authority=authority,
            )


@pytest.mark.parametrize(
    ("operation", "outcome"),
    [
        ("grid_sample", "PASS"),
        ("scatter_add", "PASS"),
        ("grid_sample", "COMPATIBILITY_FAIL"),
        ("scatter_add", "COMPATIBILITY_FAIL"),
    ],
)
def test_subprobe_validator_accepts_only_pass_or_exact_strict_failure(
    operation: str,
    outcome: str,
) -> None:
    value = _subprobe(operation, outcome=outcome)
    _, observed = contract.validate_subprobe_receipt(
        value,
        expected_operation=operation,
        expected_python=_python_identity(),
        expected_stack=_stack_identity(),
        expected_device=_device_identity(),
    )
    assert observed == outcome


@pytest.mark.parametrize(
    ("operation", "message", "stage"),
    [
        ("grid_sample", "out of memory", "grid_backward"),
        (
            "grid_sample",
            contract.EXPECTED_GRID_STRICT_ERROR,
            "grid_forward",
        ),
        (
            "scatter_add",
            contract.EXPECTED_SCATTER_STRICT_ERROR + " changed",
            "scatter_full_forward_candidate_0",
        ),
        (
            "scatter_add",
            contract.EXPECTED_SCATTER_STRICT_ERROR,
            "scatter_backward",
        ),
    ],
)
def test_subprobe_validator_rejects_unexpected_error_or_stage(
    operation: str,
    message: str,
    stage: str,
) -> None:
    value = _subprobe(
        operation,
        outcome="COMPATIBILITY_FAIL",
        message=message,
        stage=stage,
    )
    with pytest.raises(RuntimeError):
        contract.validate_subprobe_receipt(
            value,
            expected_operation=operation,
            expected_python=_python_identity(),
            expected_stack=_stack_identity(),
            expected_device=_device_identity(),
        )


def test_any_warning_is_operational_failure_even_if_kernel_is_allowlisted() -> None:
    warning = _make_warning(
        contract.EXPECTED_GRID_STRICT_ERROR.replace(
            "torch.use_deterministic_algorithms(True)",
            "torch.use_deterministic_algorithms(True, warn_only=True)",
        )
    )
    value = _subprobe("grid_sample", warnings=[warning])
    with pytest.raises(RuntimeError, match="warning"):
        contract.validate_subprobe_receipt(
            value,
            expected_operation="grid_sample",
            expected_python=_python_identity(),
            expected_stack=_stack_identity(),
            expected_device=_device_identity(),
        )


def test_subprobe_rejects_stack_device_and_determinism_drift() -> None:
    value = _subprobe("grid_sample")
    mutations = {
        "stack": {**_stack_identity(), "hip_version": "different"},
        "device": {**_device_identity(), "visible_device_name": "other"},
        "determinism": {
            **contract.DETERMINISM_CONTRACT,
            "warn_only_enabled": True,
        },
    }
    for field, changed in mutations.items():
        core = {
            key: item
            for key, item in value.items()
            if key != "content_sha256"
        }
        core[field] = changed
        with pytest.raises(PermissionError):
            contract.validate_subprobe_receipt(
                contract.with_content_sha256(core),
                expected_operation="grid_sample",
                expected_python=_python_identity(),
                expected_stack=_stack_identity(),
                expected_device=_device_identity(),
            )


@pytest.mark.parametrize(
    ("operation", "field", "key", "changed"),
    [
        (
            "grid_sample",
            "counts",
            "grid_sample_forward_invocation_count",
            19,
        ),
        (
            "scatter_add",
            "counts",
            "scatter_add_completion_count",
            147,
        ),
        (
            "grid_sample",
            "checks",
            "exact_grid_call_count",
            False,
        ),
    ],
)
def test_subprobe_rejects_exact_count_or_check_drift(
    operation: str,
    field: str,
    key: str,
    changed: object,
) -> None:
    value = _subprobe(operation)
    core = {
        name: item
        for name, item in value.items()
        if name != "content_sha256"
    }
    core[field] = {**core[field], key: changed}
    with pytest.raises(RuntimeError):
        contract.validate_subprobe_receipt(
            contract.with_content_sha256(core),
            expected_operation=operation,
            expected_python=_python_identity(),
            expected_stack=_stack_identity(),
            expected_device=_device_identity(),
        )


def test_subprobe_rejects_operation_shape_drift() -> None:
    value = _subprobe("grid_sample")
    core = {
        name: item
        for name, item in value.items()
        if name != "content_sha256"
    }
    core["operation_spec"] = {
        **contract.GRID_OPERATION,
        "grid_call_shape": [4, 4095, 1, 2],
    }
    with pytest.raises(PermissionError, match="identity"):
        contract.validate_subprobe_receipt(
            contract.with_content_sha256(core),
            expected_operation="grid_sample",
            expected_python=_python_identity(),
            expected_stack=_stack_identity(),
            expected_device=_device_identity(),
        )


def test_subprobe_rejects_python_bool_integer_equality_collisions() -> None:
    value = _subprobe("grid_sample")
    mutations = (
        (
            "counts",
            {
                **value["counts"],
                "synthetic_dense_tensor_count": True,
            },
        ),
        (
            "checks",
            {
                **value["checks"],
                "exact_grid_call_count": 1,
            },
        ),
        (
            "operation_spec",
            {
                **contract.GRID_OPERATION,
                "grid_call_shape": [4, 4096, True, 2],
            },
        ),
    )
    for field, changed in mutations:
        core = {
            name: item
            for name, item in value.items()
            if name != "content_sha256"
        }
        core[field] = changed
        with pytest.raises((PermissionError, RuntimeError)):
            contract.validate_subprobe_receipt(
                contract.with_content_sha256(core),
                expected_operation="grid_sample",
                expected_python=_python_identity(),
                expected_stack=_stack_identity(),
                expected_device=_device_identity(),
            )


def test_public_and_internal_cli_are_exact() -> None:
    digest = "a" * 64
    public = launcher.parse_args(
        ["--review-sha256", digest, "--authorization-sha256", digest]
    )
    assert vars(public) == {
        "review_sha256": digest,
        "authorization_sha256": digest,
    }
    internal = runner.parse_args(
        [
            "--run",
            "--review-sha256",
            digest,
            "--authorization-sha256",
            digest,
            "--preflight-sha256",
            digest,
        ]
    )
    assert vars(internal) == {
        "run": True,
        "review_sha256": digest,
        "authorization_sha256": digest,
        "preflight_sha256": digest,
    }
    with pytest.raises(SystemExit):
        launcher.parse_args(
            [
                "--review-sha256",
                digest,
                "--authorization-sha256",
                digest,
                "--output-root",
                "/tmp/alternate",
            ]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--run",
                "--review-sha256",
                digest,
                "--authorization-sha256",
                digest,
            ]
        )


def test_launcher_environment_is_sanitized_and_fixed_to_visible_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        launcher.PREFLIGHT_ENVIRONMENT_KEY,
    ):
        monkeypatch.setenv(name, "hostile")
    environment = launcher._launch_environment()
    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert all(environment[name] == "1" for name in launcher.THREAD_ENVIRONMENT)
    assert not any(
        name in environment
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "CUDA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "HSA_OVERRIDE_GFX_VERSION",
            launcher.PREFLIGHT_ENVIRONMENT_KEY,
        )
    )


def test_launcher_disables_bytecode_before_contract_dynamic_import() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert source.index("sys.dont_write_bytecode = True") < source.index(
        "_CONTRACT_SPEC.loader.exec_module(contract)"
    )


def test_no_tensor_preflight_subprocess_contract_is_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = {
        key: value
        for key, value in _preflight().items()
        if key
        in {
            "preflight_child_process_id",
            "python",
            "stack",
            "device",
            "tensor_allocation_count",
            "memory_allocated_bytes",
            "memory_reserved_bytes",
            "payload_open_count",
            "model_or_runtime_root_open_count",
        }
    }
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> object:
        captured["argv"] = argv
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=_canonical_raw(observation),
            stderr=b"",
        )

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher._run_no_tensor_preflight({"HIP_VISIBLE_DEVICES": "0"}) == (
        observation
    )
    assert captured["argv"][1:4] == ["-I", "-B", "-c"]
    assert captured["cwd"] == "/tmp"
    assert captured["capture_output"] is True
    assert "torch" not in sys.modules


def test_immediate_exec_passes_only_exact_internal_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    args = SimpleNamespace(review_sha256=digest, authorization_sha256=digest)
    receipt_raw = _canonical_raw(_preflight())
    observed: dict[str, object] = {}

    class ExecCalled(Exception):
        pass

    def fake_execve(
        executable: str,
        argv: list[str],
        environment: dict[str, str],
    ) -> None:
        observed.update(
            executable=executable,
            argv=argv,
            environment=environment,
        )
        raise ExecCalled

    monkeypatch.setattr(launcher.os, "execve", fake_execve)
    with pytest.raises(ExecCalled):
        launcher._exec_runner(
            args,
            receipt_raw=receipt_raw,
            environment={"HIP_VISIBLE_DEVICES": "0"},
        )
    argv = observed["argv"]
    assert isinstance(argv, list)
    assert argv[1:4] == ["-I", "-B", str(launcher.RUNNER_PATH)]
    assert argv[4:9] == [
        "--run",
        "--review-sha256",
        digest,
        "--authorization-sha256",
        digest,
    ]
    assert argv[9] == "--preflight-sha256"
    assert len(argv) == 11


def test_run_subprobe_uses_isolated_child_and_validates_canonical_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _subprobe("grid_sample")
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> object:
        captured["argv"] = argv
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=_canonical_raw(receipt),
            stderr=b"",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    preflight = _preflight()
    observed, outcome = runner._run_subprobe(
        program=runner.GRID_CHILD_PROGRAM,
        expected_operation="grid_sample",
        preflight=preflight,
    )
    assert observed == receipt
    assert outcome == "PASS"
    assert captured["argv"][1:4] == ["-I", "-B", "-c"]
    assert captured["cwd"] == "/tmp"
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert runner.PREFLIGHT_ENVIRONMENT_KEY not in environment
    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert "torch" not in sys.modules


@pytest.mark.parametrize(
    ("stderr", "stdout"),
    [
        (b"unexpected\n", b""),
        (b"", b"not-json\n"),
        (b"", _canonical_raw(_subprobe("scatter_add"))),
    ],
)
def test_run_subprobe_rejects_stderr_malformed_or_wrong_operation(
    monkeypatch: pytest.MonkeyPatch,
    stderr: bytes,
    stdout: bytes,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=stdout,
            stderr=stderr,
        ),
    )
    with pytest.raises((RuntimeError, ValueError, PermissionError)):
        runner._run_subprobe(
            program=runner.GRID_CHILD_PROGRAM,
            expected_operation="grid_sample",
            preflight=_preflight(),
        )


def test_run_subprobe_rejects_nonzero_child_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=9,
            stdout=b"",
            stderr=b"synthetic child failure\n",
        ),
    )
    with pytest.raises(RuntimeError, match="child failed"):
        runner._run_subprobe(
            program=runner.GRID_CHILD_PROGRAM,
            expected_operation="grid_sample",
            preflight=_preflight(),
        )


def _patch_parent_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    subprobes: list[tuple[dict[str, object], str] | BaseException],
) -> tuple[dict[str, object], dict[str, str], dict[str, object], bytes]:
    (tmp_path / ".generated").mkdir()
    authority = _source_authority()
    sources = _sources()
    preflight = _preflight(authority)
    preflight_raw = _canonical_raw(preflight)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_load_source_authority",
        lambda **kwargs: (authority, sources),
    )
    monkeypatch.setattr(
        runner,
        "_validate_preflight_from_environment",
        lambda **kwargs: (preflight, preflight_raw),
    )
    queue = list(subprobes)

    def fake_subprobe(**kwargs: object) -> tuple[dict[str, object], str]:
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(runner, "_run_subprobe", fake_subprobe)
    return authority, sources, preflight, preflight_raw


@pytest.mark.parametrize(
    ("grid_outcome", "scatter_outcome", "expected_exit", "expected_status"),
    [
        ("PASS", "PASS", 0, contract.RESULT_PASS),
        (
            "COMPATIBILITY_FAIL",
            "PASS",
            10,
            contract.RESULT_COMPATIBILITY_FAIL,
        ),
        (
            "PASS",
            "COMPATIBILITY_FAIL",
            10,
            contract.RESULT_COMPATIBILITY_FAIL,
        ),
        (
            "COMPATIBILITY_FAIL",
            "COMPATIBILITY_FAIL",
            10,
            contract.RESULT_COMPATIBILITY_FAIL,
        ),
    ],
)
def test_full_mocked_terminal_pass_and_compatibility_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    grid_outcome: str,
    scatter_outcome: str,
    expected_exit: int,
    expected_status: str,
) -> None:
    root = tmp_path / contract.OUTPUT_ROOT_RELATIVE_PATH
    grid = _subprobe("grid_sample", outcome=grid_outcome)
    scatter = _subprobe("scatter_add", outcome=scatter_outcome)
    _patch_parent_inputs(
        monkeypatch,
        tmp_path,
        subprobes=[(grid, grid_outcome), (scatter, scatter_outcome)],
    )
    try:
        observed = runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
            preflight_file_sha256="c" * 64,
        )
        assert observed == expected_exit
        assert sorted(path.name for path in root.iterdir()) == [
            "access.json",
            "completed.json",
            "reservation.json",
            "result.json",
        ]
        result = contract.parse_canonical_json(
            (root / "result.json").read_bytes(),
            name="result",
        )
        assert result["status"] == expected_status
        contract.validate_result_receipt(result)
        completion = contract.parse_canonical_json(
            (root / "completed.json").read_bytes(),
            name="completion",
        )
        contract.validate_completion_receipt(completion)
        assert completion["retry_authorized"] is False
        assert completion["downstream_denials"] == contract.DOWNSTREAM_DENIALS
        assert stat.S_IMODE(root.stat().st_mode) == 0o555
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o444
            for path in root.iterdir()
        )
    finally:
        _unseal(root)


def test_post_reservation_child_operational_failure_terminalizes_no_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / contract.OUTPUT_ROOT_RELATIVE_PATH
    grid = _subprobe("grid_sample")
    _patch_parent_inputs(
        monkeypatch,
        tmp_path,
        subprobes=[(grid, "PASS"), RuntimeError("unexpected child stderr")],
    )
    try:
        observed = runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
            preflight_file_sha256="c" * 64,
        )
        assert observed == contract.EXIT_OPERATIONAL_FAILURE
        assert sorted(path.name for path in root.iterdir()) == [
            "failed.json",
            "reservation.json",
        ]
        failed = contract.parse_canonical_json(
            (root / "failed.json").read_bytes(),
            name="failure",
        )
        contract.validate_failure_receipt(failed)
        assert failed["stage"] == "scatter_add_subprobe"
        assert failed["compatibility_result"] is None
        assert failed["retry_authorized"] is False
        assert failed["v3_probe_root"]["inspected"] is False
        assert stat.S_IMODE(root.stat().st_mode) == 0o555
    finally:
        _unseal(root)


def test_failure_before_reservation_receipt_still_seals_consumed_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / contract.OUTPUT_ROOT_RELATIVE_PATH
    _patch_parent_inputs(
        monkeypatch,
        tmp_path,
        subprobes=[
            (_subprobe("grid_sample"), "PASS"),
            (_subprobe("scatter_add"), "PASS"),
        ],
    )
    original = runner._publish_json
    injected = {"done": False}

    def inject(
        root_fd: int,
        name: str,
        value: dict[str, object],
    ) -> tuple[dict[str, object], bytes]:
        if name == "reservation.json" and not injected["done"]:
            injected["done"] = True
            raise OSError("injected reservation publication failure")
        return original(root_fd, name, value)

    monkeypatch.setattr(runner, "_publish_json", inject)
    try:
        assert runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
            preflight_file_sha256="c" * 64,
        ) == contract.EXIT_OPERATIONAL_FAILURE
        assert [path.name for path in root.iterdir()] == ["failed.json"]
        failed = contract.parse_canonical_json(
            (root / "failed.json").read_bytes(),
            name="failure",
        )
        assert failed["attempt_consumed"] is True
        assert failed["durable_prefix"] == []
        assert failed["stage"] == "reservation_publication"
        assert stat.S_IMODE(root.stat().st_mode) == 0o555
    finally:
        _unseal(root)


def test_failure_after_root_mkdir_before_reserve_return_terminalizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / contract.OUTPUT_ROOT_RELATIVE_PATH
    _patch_parent_inputs(
        monkeypatch,
        tmp_path,
        subprobes=[
            (_subprobe("grid_sample"), "PASS"),
            (_subprobe("scatter_add"), "PASS"),
        ],
    )
    original_fsync = runner.os.fsync
    calls = {"count": 0}

    def fail_first_fsync(descriptor: int) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("injected post-mkdir reservation failure")
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", fail_first_fsync)
    try:
        assert runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
            preflight_file_sha256="c" * 64,
        ) == contract.EXIT_OPERATIONAL_FAILURE
        assert [path.name for path in root.iterdir()] == ["failed.json"]
        failed = contract.parse_canonical_json(
            (root / "failed.json").read_bytes(),
            name="post-mkdir reservation failure",
        )
        contract.validate_failure_receipt(failed)
        assert failed["attempt_consumed"] is True
        assert failed["durable_prefix"] == []
        assert failed["stage"] == "reservation_initialization"
        assert failed["retry_authorized"] is False
        assert stat.S_IMODE(root.stat().st_mode) == 0o555
    finally:
        _unseal(root)


def test_failure_after_result_binds_durable_prefix_and_has_no_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / contract.OUTPUT_ROOT_RELATIVE_PATH
    _patch_parent_inputs(
        monkeypatch,
        tmp_path,
        subprobes=[
            (_subprobe("grid_sample"), "PASS"),
            (_subprobe("scatter_add"), "PASS"),
        ],
    )
    original = runner._publish_json

    def inject(
        root_fd: int,
        name: str,
        value: dict[str, object],
    ) -> tuple[dict[str, object], bytes]:
        if name == "completed.json":
            raise OSError("injected completion publication failure")
        return original(root_fd, name, value)

    monkeypatch.setattr(runner, "_publish_json", inject)
    try:
        assert runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
            preflight_file_sha256="c" * 64,
        ) == contract.EXIT_OPERATIONAL_FAILURE
        assert sorted(path.name for path in root.iterdir()) == [
            "access.json",
            "failed.json",
            "reservation.json",
            "result.json",
        ]
        failed = contract.parse_canonical_json(
            (root / "failed.json").read_bytes(),
            name="failure",
        )
        assert failed["stage"] == "completion_publication"
        assert [row["path"] for row in failed["durable_prefix"]] == [
            "access.json",
            "reservation.json",
            "result.json",
        ]
        assert failed["compatibility_result"] is None
    finally:
        _unseal(root)


def test_output_root_reservation_is_one_attempt_and_never_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".generated").mkdir()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output, root_fd, parent_fd = runner._reserve_output_root()
    os.close(root_fd)
    os.close(parent_fd)
    assert output == tmp_path / contract.OUTPUT_ROOT_RELATIVE_PATH
    with pytest.raises(FileExistsError):
        runner._reserve_output_root()


def test_sealing_rejects_foreign_or_symlink_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".generated").mkdir()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    root, root_fd, parent_fd = runner._reserve_output_root()
    try:
        (root / "foreign").symlink_to("/tmp")
        with pytest.raises(PermissionError, match="unexpected"):
            runner._seal_terminal(root_fd, parent_fd)
    finally:
        os.close(root_fd)
        os.close(parent_fd)


def test_result_and_completion_never_grant_downstream_authority() -> None:
    reservation = _binding("reservation.json", "6")
    access = _binding("access.json", "8")
    preflight = _preflight()
    grid = _subprobe("grid_sample")
    scatter = _subprobe("scatter_add")
    result = runner._result_receipt(
        attempt_identity="a" * 64,
        reservation_binding=reservation,
        access_binding=access,
        preflight=preflight,
        grid=grid,
        grid_outcome="PASS",
        scatter=scatter,
        scatter_outcome="PASS",
    )
    assert result["status"] == contract.RESULT_PASS
    assert result["scientific_metric"] is None
    assert result["checkpoint_qualified"] is False
    assert all(value is False for value in result["downstream_denials"].values())
    result_raw = _canonical_raw(result)
    result_binding = contract.artifact_binding(
        "result.json",
        result_raw,
        content_sha256=result["content_sha256"],
    )
    completion = runner._completion_receipt(
        attempt_identity="a" * 64,
        reservation_binding=reservation,
        access_binding=access,
        result_binding=result_binding,
        result_status=result["status"],
    )
    assert completion["status"] == contract.COMPLETION_PASS
    assert completion["retry_authorized"] is False
    assert all(
        value is False for value in completion["downstream_denials"].values()
    )


def test_compatibility_fail_is_result_not_operational_failure() -> None:
    reservation = _binding("reservation.json", "6")
    access = _binding("access.json", "8")
    grid = _subprobe("grid_sample", outcome="COMPATIBILITY_FAIL")
    scatter = _subprobe("scatter_add")
    result = runner._result_receipt(
        attempt_identity="a" * 64,
        reservation_binding=reservation,
        access_binding=access,
        preflight=_preflight(),
        grid=grid,
        grid_outcome="COMPATIBILITY_FAIL",
        scatter=scatter,
        scatter_outcome="PASS",
    )
    assert result["status"] == contract.RESULT_COMPATIBILITY_FAIL
    assert result["subprobe_outcomes"] == {
        "grid_sample": "COMPATIBILITY_FAIL",
        "scatter_add": "PASS",
    }
    assert result["checkpoint_qualified"] is False
    assert contract.EXIT_COMPATIBILITY_FAIL != contract.EXIT_OPERATIONAL_FAILURE


def test_attempt_identity_binds_preflight_authority_operations_and_root() -> None:
    authority = _source_authority()
    first = _preflight(authority)
    identity = contract.make_attempt_identity(
        source_authority=authority,
        preflight=first,
    )
    assert contract.is_sha256(identity)
    core = {
        key: item
        for key, item in first.items()
        if key != "content_sha256"
    }
    core["stack"] = {**first["stack"], "hip_version": "changed"}
    changed = contract.with_content_sha256(core)
    assert identity != contract.make_attempt_identity(
        source_authority=authority,
        preflight=changed,
    )
    identity_core = contract.attempt_identity_core(
        source_authority=authority,
        preflight=first,
    )
    assert identity_core["output_root"] == contract.OUTPUT_ROOT_RELATIVE_PATH
    assert identity_core["attempt_index"] == 1
    assert identity_core["maximum_attempts"] == 1
    assert identity_core["operation_contract_sha256"] == (
        contract.OPERATION_CONTRACT_SHA256
    )


def test_prohibited_access_and_training_counts_are_exact_zero() -> None:
    assert contract.PROHIBITED_OPEN_COUNTS
    assert contract.ZERO_TRAINING_COUNTS
    assert all(value == 0 for value in contract.PROHIBITED_OPEN_COUNTS.values())
    assert all(value == 0 for value in contract.ZERO_TRAINING_COUNTS.values())
    assert contract.OUTPUT_CONTRACT["result_contains_tensors"] is False
    assert contract.OUTPUT_CONTRACT["result_contains_scientific_metrics"] is False
    assert contract.OUTPUT_CONTRACT["one_attempt_no_retry"] is True


def test_no_payload_or_model_read_api_is_present_in_child_programs() -> None:
    forbidden = (
        "torch.load",
        "Image.open",
        "numpy",
        "np.load",
        "open(",
        "Path(",
        "checkpoint",
        "dataset",
        "rgb_byte",
        "rgb_decode",
        "sealed",
        "optimizer.step",
    )
    combined = runner.GRID_CHILD_PROGRAM + runner.SCATTER_CHILD_PROGRAM
    for token in forbidden:
        assert token not in combined


def test_only_review_and_authorization_are_runtime_placeholders() -> None:
    assert contract.REVIEW_RELATIVE_PATH.endswith(
        "_r9700_strict_compatibility_v1_source_review_2026-07-24.json"
    )
    assert contract.AUTHORIZATION_RELATIVE_PATH.endswith(
        "_r9700_strict_compatibility_v1_execution_authorization_2026-07-24.json"
    )
    assert contract.REVIEW_RELATIVE_PATH not in contract.SOURCE_PATHS
    assert contract.AUTHORIZATION_RELATIVE_PATH not in contract.SOURCE_PATHS
