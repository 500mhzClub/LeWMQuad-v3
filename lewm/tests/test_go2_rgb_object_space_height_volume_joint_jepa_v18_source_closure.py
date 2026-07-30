from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "scripts/check_go2_rgb_object_space_height_volume_joint_jepa_v18_"
    "source_closure.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_IMPORTED_BEFORE = set(sys.modules)
checker = _load("_test_v18_height_volume_source_closure", CHECKER_PATH)
_IMPORTED_BY_CHECKER = set(sys.modules) - _IMPORTED_BEFORE


def test_checker_import_is_source_only_and_uses_fresh_v18_identity() -> None:
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in _IMPORTED_BY_CHECKER
        for prefix in ("torch", "numpy", "PIL", "lewm", "scripts")
    )
    assert checker.SCHEMA == (
        "lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_source_manifest"
    )
    assert checker._V13._BASE.SCHEMA == checker.SCHEMA
    assert checker.MANIFEST_PATH.relative_to(ROOT).as_posix() == (
        "docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_"
        "source_manifest_2026-07-30.json"
    )


def test_entrypoints_and_dynamic_private_bases_are_registered() -> None:
    assert checker.ENTRYPOINTS == (
        (
            "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
            "v18_object_space_height_volume.py"
        ),
        "scripts/run_go2_rgb_object_space_height_volume_joint_jepa_v18.py",
        "scripts/execute_go2_rgb_object_space_height_volume_joint_jepa_v18.py",
        "scripts/launch_go2_rgb_object_space_height_volume_joint_jepa_v18.py",
    )
    assert {
        (
            "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
            "camera_evidence_bottleneck.py"
        ),
        "scripts/execute_go2_rgb_unified_ray_survival_joint_jepa_v14.py",
        (
            "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_"
            "v13_camera_evidence_bottleneck.py"
        ),
        (
            "scripts/launch_go2_rgb_swept_progress_survival_joint_jepa_"
            "v13_camera_evidence_bottleneck.py"
        ),
    }.issubset(set(checker.FORCED_DYNAMIC_SOURCES))


def test_candidate_is_complete_unique_and_contains_no_runtime_material() -> None:
    manifest = checker.build_manifest()
    assert manifest["schema"] == checker.SCHEMA
    assert manifest["source_count"] == len(manifest["source_paths"])
    assert manifest["source_count"] == len(manifest["source_bindings"])
    assert len(manifest["source_paths"]) == len(set(manifest["source_paths"]))
    assert manifest["source_paths"] == [
        binding["path"] for binding in manifest["source_bindings"]
    ]
    assert manifest["generated_or_runtime_artifact_open_count"] == 0
    assert manifest["dataset_or_rgb_open_count"] == 0
    assert manifest["tensor_checkpoint_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0
    assert manifest["whole_tree_export_authorized"] is False
    if checker.MANIFEST_PATH.is_file():
        checker.verify_manifest(require_tracked=False)


def test_discovered_closure_is_safe_python_without_ignore_bypass() -> None:
    paths = set(checker.discover_source_closure())
    assert set(checker.ENTRYPOINTS).issubset(paths)
    assert set(checker.FORCED_DYNAMIC_SOURCES).issubset(paths)
    for relative in paths:
        checker._safe_source_path(relative)
        path = Path(relative)
        folded = tuple(part.casefold() for part in path.parts)
        assert path.suffix == ".py"
        assert not set(folded).intersection(checker.FORBIDDEN_PATH_PARTS)
        assert not any(part.startswith(("sealed", "heldout", "held_out")) for part in folded)
    source = CHECKER_PATH.read_text(encoding="utf-8")
    assert 'mode.add_argument("--emit"' in source
    assert 'mode.add_argument("--write"' in source
    assert 'parser.add_argument("--require-tracked"' in source
    assert "--no-ignore" not in source
