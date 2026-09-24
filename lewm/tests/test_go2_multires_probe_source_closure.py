from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "scripts/check_go2_multires_probe_source_closure.py"
SPEC = importlib.util.spec_from_file_location(
    "_test_go2_multires_probe_source_closure",
    CHECKER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_recursive_manifest_matches_every_discovered_source_byte() -> None:
    checker.verify_manifest(require_tracked=False)
    manifest = checker.build_manifest()
    assert manifest["source_count"] == len(manifest["source_paths"])
    assert manifest["source_count"] == len(manifest["source_bindings"])
    assert len(manifest["source_paths"]) == len(set(manifest["source_paths"]))
    assert manifest["source_paths"] == [
        binding["path"] for binding in manifest["source_bindings"]
    ]
    assert manifest["generated_input_open_count"] == 0
    assert manifest["tensor_checkpoint_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0
    assert manifest["whole_tree_export_authorized"] is False


def test_dynamic_runtime_edges_and_new_mechanism_are_inside_closure() -> None:
    paths = set(checker.discover_source_closure())
    required = {
        *checker.ENTRYPOINTS,
        *checker.FORCED_DYNAMIC_SOURCES,
        "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py",
        (
            "lewm/models/shared_observable_camera_ray_jepa_v5_"
            "protected_camera_adaptation_v4_tail_depth.py"
        ),
        "lewm/models/shared_observable_camera_ray_jepa_v5.py",
        "lewm/models/observable_camera_ray_evidence_v4.py",
        "lewm/models/observable_camera_ray_evidence_v4_training.py",
        "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py",
    }
    assert required.issubset(paths)
    assert not any(
        path.startswith(checker.FORBIDDEN_RUNNER_PREFIX)
        for path in paths
    )


def test_closure_contains_only_safe_python_paths() -> None:
    for relative in checker.discover_source_closure():
        checker._safe_source_path(relative)
        path = Path(relative)
        assert path.suffix == ".py"
        assert ".generated" not in path.parts
        assert not any(part.startswith("sealed") for part in path.parts)


def test_discovery_honors_ignores_and_source_reads_reject_symlinks(
    tmp_path: Path,
) -> None:
    source = CHECKER_PATH.read_text(encoding="utf-8")
    assert ".rglob(" not in source
    assert '"rg",\n            "--files"' in source
    assert "os.O_NOFOLLOW" in source
    target = tmp_path / "source.py"
    target.write_text("value = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    with pytest.raises((OSError, PermissionError)):
        checker._read_regular_source(link)
