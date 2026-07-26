from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "scripts/check_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_source_closure.py"
)


def _load_checker(name: str = "_masked_pair_tubelet_v11_closure_test"):
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_source_checker_import_is_stdlib_only() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_v11_closure", {str(CHECKER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert module.REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(
    module.contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
)
print("PASS")
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


def test_recursive_closure_contains_every_v11_dynamic_source() -> None:
    checker = _load_checker()
    discovered = set(checker.discover_source_closure())
    assert checker.REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(discovered)
    assert set(checker.contract.ADDITIVE_SOURCE_PATHS).issubset(discovered)


def test_manifest_builder_is_canonical_and_runtime_free() -> None:
    checker = _load_checker("_masked_pair_tubelet_v11_manifest_test")
    manifest = checker.build_manifest()
    assert manifest["status"] == "PASS_SOURCE_CLOSURE"
    assert manifest["generated_input_open_count"] == 0
    assert manifest["checkpoint_or_tensor_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0
    assert manifest["source_count"] == len(manifest["source_paths"])
    assert manifest["source_paths"] == sorted(manifest["source_paths"])
    assert checker.contract.validate_source_manifest(
        checker.contract.canonical_json_bytes(manifest) + b"\n"
    ) == manifest


@pytest.mark.parametrize(
    "relative",
    (
        "sealed/payload.py",
        "heldout/payload.py",
        ".generated/runtime.py",
        "config/sealed_test.json",
    ),
)
def test_source_checker_rejects_protected_or_runtime_paths(relative: str) -> None:
    checker = _load_checker(f"_masked_pair_tubelet_v11_reject_{len(relative)}")
    with pytest.raises(PermissionError):
        checker._safe_source_path(relative)
