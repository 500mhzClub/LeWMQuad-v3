from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "scripts/check_go2_direct_egocentric_bev_state_jepa_v1_source_closure.py"
)
LAUNCHER_PATH = (
    ROOT / "scripts/launch_go2_direct_egocentric_bev_state_jepa_v1.py"
)


def _load_checker(name: str = "_direct_bev_v1_closure_test"):
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
spec = importlib.util.spec_from_file_location("_direct_bev_v1_closure", {str(CHECKER_PATH)!r})
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


def test_launcher_import_is_source_only_and_exactly_rebound() -> None:
    program = f"""
import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("_direct_bev_v1_launcher", {str(LAUNCHER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert module._V11.contract is module.contract
assert module._V11._BASE.contract is module.contract
assert module._V11._BASE.RUNNER_PATH == module.ROOT / module.contract.RUNNER_RELATIVE_PATH
assert module._V11.PREFLIGHT_ENVIRONMENT_KEY == module.PREFLIGHT_ENVIRONMENT_KEY
assert module._V11._BASE.PREFLIGHT_ENVIRONMENT_KEY == module.PREFLIGHT_ENVIRONMENT_KEY
assert Path(module._V11.__file__).resolve() == Path(module.__file__).resolve()
assert Path(module._V11._BASE.__file__).resolve() == Path(module.__file__).resolve()
args = module.parse_args([
    "--review-sha256", "0" * 64,
    "--authorization-sha256", "1" * 64,
])
assert args.review_sha256 == "0" * 64
assert args.authorization_sha256 == "1" * 64
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


def test_recursive_closure_contains_every_direct_dynamic_source() -> None:
    checker = _load_checker()
    discovered = set(checker.discover_source_closure())
    assert checker.REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(discovered)
    assert set(
        checker.contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
    ).issubset(discovered)
    assert set(checker.contract.ADDITIVE_SOURCE_PATHS).issubset(discovered)


def test_manifest_builder_is_canonical_and_runtime_free() -> None:
    checker = _load_checker("_direct_bev_v1_manifest_test")
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
    checker = _load_checker(f"_direct_bev_v1_reject_{len(relative)}")
    with pytest.raises(PermissionError):
        checker._safe_source_path(relative)
