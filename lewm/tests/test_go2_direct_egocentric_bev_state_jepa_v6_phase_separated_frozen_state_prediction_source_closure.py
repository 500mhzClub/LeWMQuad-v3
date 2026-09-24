from __future__ import annotations

import copy
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "scripts/check_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction_source_closure.py"
)


def _load_checker(name: str = "_direct_bev_v6_phase_closure_test"):
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_source_checker_import_is_stdlib_only_and_exactly_rebound() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location(
    '_direct_bev_v6_phase_closure', {str(CHECKER_PATH)!r}
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name.startswith('torch.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert module.REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    module.contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)
assert module._V5.contract is module.contract
assert module._V5._V4.contract is module.contract
assert module._V5._V4._V3.contract is module.contract
assert module._V5._V4._V3._V2.contract is module.contract
assert module._V5._V4._V3._V2._V1.contract is module.contract
assert module._V5._V4._V3._V2._V1._V11.contract is module.contract
assert module._V5._V4._V3._V2._V1._V11._V10.contract is module.contract
backend = module._V5._V4._V3._V2._V1._V11._V10._BASE
assert backend.ENTRYPOINTS == tuple(module.contract.SOURCE_MANIFEST_ENTRYPOINTS)
assert backend.FORCED_DYNAMIC_SOURCES == tuple(
    module.contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)
assert len(module.contract.ADDITIVE_SOURCE_PATHS) == 10
assert len(module.contract.SOURCE_PATHS) == 111
print('PASS')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PASS\n"
    assert completed.stderr == ""


def test_recursive_closure_contains_exact_v6_source_surface() -> None:
    checker = _load_checker()
    discovered = set(checker.discover_source_closure())
    assert checker.REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(discovered)
    assert set(
        checker.contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
    ).issubset(discovered)
    assert set(checker.contract.ADDITIVE_SOURCE_PATHS) == (
        checker.V6_ADDITIVE_SOURCE_PATHS
    )
    assert len(checker.contract.ADDITIVE_SOURCE_PATHS) == 10
    assert len(checker.contract.SOURCE_PATHS) == 111


def test_candidate_is_canonical_runtime_free_and_manifest_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _load_checker("_direct_bev_v6_phase_prepare_test")
    backend = checker._V5._V4._V3._V2._V1._V11._V10
    original_read = backend._read_regular_source
    manifest_path = (
        checker.ROOT / checker.contract.SOURCE_MANIFEST_RELATIVE_PATH
    ).resolve()
    opened: list[Path] = []

    def guarded_read(source_file: Path) -> bytes:
        resolved = Path(source_file).resolve()
        assert resolved != manifest_path
        opened.append(resolved)
        return original_read(source_file)

    monkeypatch.setattr(backend, "_read_regular_source", guarded_read)
    manifest = checker.build_manifest()
    assert opened
    assert manifest["status"] == "PASS_SOURCE_CLOSURE"
    assert manifest["generated_input_open_count"] == 0
    assert manifest["checkpoint_or_tensor_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0
    assert manifest["whole_tree_export_authorized"] is False
    assert manifest["source_count"] == 111
    assert manifest["source_paths"] == sorted(manifest["source_paths"])
    assert checker.contract.validate_source_manifest(
        checker.contract.canonical_json_bytes(manifest) + b"\n"
    ) == manifest


def test_strict_verification_accepts_only_current_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _load_checker("_direct_bev_v6_phase_verify_test")
    backend = checker._V5._V4._V3._V2._V1._V11._V10
    original_read = backend._read_regular_source
    candidate = checker.build_manifest()
    candidate_raw = checker.contract.canonical_json_bytes(candidate) + b"\n"
    manifest_path = (
        checker.ROOT / checker.contract.SOURCE_MANIFEST_RELATIVE_PATH
    ).resolve()

    def candidate_read(source_file: Path) -> bytes:
        if Path(source_file).resolve() == manifest_path:
            return candidate_raw
        return original_read(source_file)

    monkeypatch.setattr(backend, "_read_regular_source", candidate_read)
    assert checker.verify_manifest() == candidate

    stale = copy.deepcopy(candidate)
    stale["source_bindings"][0]["file_sha256"] = "0" * 64
    stale["source_bindings_sha256"] = checker.contract.canonical_json_sha256(
        stale["source_bindings"]
    )
    stale.pop("content_sha256")
    stale = checker.contract.with_content_sha256(stale)
    stale_raw = checker.contract.canonical_json_bytes(stale) + b"\n"

    def stale_read(source_file: Path) -> bytes:
        if Path(source_file).resolve() == manifest_path:
            return stale_raw
        return original_read(source_file)

    monkeypatch.setattr(backend, "_read_regular_source", stale_read)
    with pytest.raises(RuntimeError, match="source manifest changed"):
        checker.verify_manifest()


def test_manifest_writer_remains_exclusive_and_protected_paths_fail() -> None:
    checker = _load_checker("_direct_bev_v6_phase_exclusive_test")
    assert checker._write_manifest_exclusive is (
        checker._V5._write_manifest_exclusive
    )
    source = inspect.getsource(checker._write_manifest_exclusive)
    assert "os.O_EXCL" in source
    assert '"O_NOFOLLOW"' in source
    assert "os.fsync" in source
    for relative in (
        "sealed/payload.py",
        "heldout/payload.py",
        ".generated/runtime.py",
        "config/sealed_test.json",
    ):
        with pytest.raises(PermissionError):
            checker._safe_source_path(relative)
