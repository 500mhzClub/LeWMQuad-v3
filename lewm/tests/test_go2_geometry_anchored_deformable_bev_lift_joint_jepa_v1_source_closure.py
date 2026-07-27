from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "scripts/"
    "check_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v1_"
    "source_closure.py"
)


def _load(name: str = "_geometry_anchored_joint_jepa_closure_test"):
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_checker_import_is_source_only() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_closure", {str(CHECKER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
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


def test_recursive_closure_is_exact_direct_base_plus_ten_sources() -> None:
    checker = _load()
    discovered = set(checker.discover_source_closure())
    assert discovered == set(checker.contract.SOURCE_PATHS)
    assert discovered == checker.EXPECTED_SOURCE_PATHS
    assert checker.REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(discovered)
    assert len(checker.ADDITIVE_SOURCE_PATHS) == 10


def test_manifest_builder_is_canonical_and_runtime_free() -> None:
    checker = _load("_geometry_anchored_joint_jepa_manifest_test")
    manifest = checker.build_manifest()
    assert manifest["status"] == "PASS_SOURCE_CLOSURE"
    assert manifest["generated_input_open_count"] == 0
    assert manifest["checkpoint_or_tensor_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0
    assert manifest["whole_tree_export_authorized"] is False
    assert manifest["source_count"] == len(checker.contract.SOURCE_PATHS)
    raw = checker.contract.canonical_json_bytes(manifest) + b"\n"
    assert checker.contract.validate_source_manifest(raw, ROOT) == manifest


@pytest.mark.parametrize(
    "relative",
    (
        "sealed/payload.py",
        "sealed_future/payload.py",
        "heldout/payload.py",
        "heldout_future/payload.py",
        ".generated/runtime.py",
        "config/sealed_test.json",
    ),
)
def test_checker_rejects_runtime_or_protected_paths(relative: str) -> None:
    checker = _load(f"_geometry_anchored_joint_jepa_reject_{len(relative)}")
    with pytest.raises(PermissionError):
        checker._safe_source_path(relative)
