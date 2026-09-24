from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKER = (
    ROOT
    / "scripts/check_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_source_closure.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "_masked_spatial_v1_source_closure_test_subject", CHECKER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_closure_is_exact_safe_and_reproducible() -> None:
    checker = _module()
    first = checker.build_manifest()
    second = checker.build_manifest()
    assert first == second
    paths = first["source_paths"]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths)) == first["source_count"]
    assert set(checker.ENTRYPOINTS).issubset(paths)
    assert set(checker.FORCED_DYNAMIC_SOURCES).issubset(paths)
    assert set(checker.EXACT_DATASET_SOURCES).issubset(paths)
    assert first["execution_authorized"] is False
    assert first["whole_tree_export_authorized"] is False
    for relative in paths:
        folded = relative.casefold()
        assert ".generated/" not in folded
        assert "sealed" not in folded
        assert "heldout" not in folded


def test_source_closure_supports_isolated_entrypoint_imports(tmp_path: Path) -> None:
    checker = _module()
    manifest = checker.build_manifest()
    for relative in manifest["source_paths"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    modules = tuple(
        relative.removesuffix(".py").replace("/", ".")
        for relative in checker.ENTRYPOINTS
    )
    code = (
        "import importlib,sys;"
        f"sys.path.insert(0,{str(tmp_path)!r});"
        f"[importlib.import_module(name) for name in {modules!r}]"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
