from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "scripts/check_go2_rgb_overlapping_tokenization_v1_source_closure.py"
)


def _load_checker(name: str):
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_checker_import_is_source_only() -> None:
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {"torch", "numpy", "PIL"}:
            raise AssertionError(f"source-only checker imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded):
        checker = _load_checker("_overlap_closure_source_only")
    assert checker.MANIFEST_PATH == (
        checker.ROOT / checker.contract.SOURCE_MANIFEST_RELATIVE_PATH
    )
    assert checker._WALKER.ENTRYPOINTS == checker.ENTRYPOINTS
    assert (
        checker._WALKER.FORCED_DYNAMIC_SOURCES
        == checker.FORCED_DYNAMIC_SOURCES
    )


def test_recursive_candidate_covers_every_discovered_source_byte() -> None:
    checker = _load_checker("_overlap_closure_candidate")
    manifest_existed_before = checker.MANIFEST_PATH.exists()
    manifest = checker.build_manifest()
    # Candidate construction is read-only.  Once a later source-freeze step
    # commits a manifest, the same test validates it without recreating it.
    assert checker.MANIFEST_PATH.exists() is manifest_existed_before
    if manifest_existed_before:
        checker.verify_manifest(require_tracked=False)
    assert manifest["schema"] == checker.SCHEMA
    assert manifest["source_count"] == len(manifest["source_paths"])
    assert manifest["source_count"] == len(manifest["source_bindings"])
    assert len(manifest["source_paths"]) == len(
        set(manifest["source_paths"])
    )
    assert manifest["source_paths"] == [
        binding["path"] for binding in manifest["source_bindings"]
    ]
    assert manifest["generated_input_open_count"] == 0
    assert manifest["tensor_checkpoint_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0
    assert manifest["whole_tree_export_authorized"] is False


def test_additive_static_and_corrected_lifecycle_edges_are_in_closure() -> None:
    checker = _load_checker("_overlap_closure_edges")
    paths = set(checker.discover_source_closure())
    required = {
        *checker.ENTRYPOINTS,
        *checker.FORCED_DYNAMIC_SOURCES,
        *checker.contract.ADDITIVE_SOURCE_PATHS,
        checker.CONTRACT_RELATIVE_PATH,
        checker.MOTION_CHECKER_RELATIVE_PATH,
        "lewm/benchmarks/go2_rgb_causal_motion_alignment_v1.py",
        "scripts/run_go2_rgb_causal_motion_alignment_v1.py",
        "scripts/launch_go2_rgb_causal_motion_alignment_v1.py",
        "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py",
        "scripts/run_go2_rgb_causal_temporal_perception_v1.py",
        "scripts/launch_go2_rgb_causal_temporal_perception_v1.py",
        "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py",
        "lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py",
        (
            "lewm/models/shared_observable_camera_ray_jepa_v5_multires_"
            "overlapping_tokenization_v1.py"
        ),
        "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py",
        "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py",
        "scripts/run_go2_shared_jepa_v5_matched_training_v1.py",
        (
            "lewm/models/shared_observable_camera_ray_jepa_v5_"
            "protected_camera_adaptation_v4_tail_depth.py"
        ),
        "lewm/models/shared_observable_camera_ray_jepa_v5.py",
        "lewm/models/observable_camera_ray_evidence_v4.py",
        "lewm/models/observable_camera_ray_evidence_v4_training.py",
    }
    assert required.issubset(paths)
    assert not any(
        path.startswith(checker._WALKER.FORBIDDEN_RUNNER_PREFIX)
        for path in paths
    )


def test_closure_contains_only_safe_regular_python_paths() -> None:
    checker = _load_checker("_overlap_closure_safe_paths")
    for relative in checker.discover_source_closure():
        checker._safe_source_path(relative)
        path = checker.ROOT / relative
        assert path.suffix == ".py"
        assert not path.is_symlink()
        assert ".generated" not in path.parts
        assert "checkpoints" not in path.parts
        assert not any(part.startswith("sealed") for part in path.parts)


@pytest.mark.parametrize(
    "relative",
    (
        ".generated/runtime/source.py",
        "lewm/sealed/private.py",
        "lewm/sealed_legacy/private.py",
        "lewm/checkpoints/state.py",
        "../escape.py",
        "/absolute/source.py",
        "lewm/models/not_python.json",
        (
            "scripts/run_go2_shared_jepa_v5_"
            "protected_camera_adaptation_v99.py"
        ),
    ),
)
def test_unsafe_source_paths_are_rejected(relative: str) -> None:
    checker = _load_checker("_overlap_closure_reject_unsafe")
    with pytest.raises(PermissionError, match="forbidden source-closure path"):
        checker._safe_source_path(relative)


def test_source_reads_reject_symlinks(tmp_path: Path) -> None:
    checker = _load_checker("_overlap_closure_symlink")
    target = tmp_path / "source.py"
    target.write_text("value = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    with pytest.raises((OSError, PermissionError)):
        checker._read_regular_source(link)
