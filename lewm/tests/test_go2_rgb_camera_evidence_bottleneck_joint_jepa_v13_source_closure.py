from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "scripts/check_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "source_closure.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_test_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_source_closure",
    CHECKER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_candidate_accounts_for_every_discovered_source_byte() -> None:
    imported_before = {
        name
        for name in sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "lewm"
        or name.startswith("lewm.")
        or name == "scripts"
        or name.startswith("scripts.")
        or name == "lewm_worlds"
        or name.startswith("lewm_worlds.")
    }
    manifest = checker.build_manifest()
    imported_after = {
        name
        for name in sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "lewm"
        or name.startswith("lewm.")
        or name == "scripts"
        or name.startswith("scripts.")
        or name == "lewm_worlds"
        or name.startswith("lewm_worlds.")
    }
    assert imported_after == imported_before
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


def test_all_entrypoints_and_forced_dynamic_edges_are_in_closure() -> None:
    paths = set(checker.discover_source_closure())
    assert set(checker.ENTRYPOINTS).issubset(paths)
    assert set(checker.FORCED_DYNAMIC_SOURCES).issubset(paths)
    assert {
        (
            "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
            "v13_camera_evidence_bottleneck.py"
        ),
        "lewm/models/shared_observable_camera_ray_jepa_v5.py",
        "lewm/models/observable_camera_ray_evidence_v4.py",
        "lewm/models/observable_camera_ray_evidence_v4_training.py",
        "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py",
        "lewm_worlds/lewm_worlds/__init__.py",
        "lewm_worlds/lewm_worlds/manifest.py",
        "lewm_worlds/lewm_worlds/families.py",
        "scripts/run_go2_shared_jepa_v5_matched_training_v1.py",
    }.issubset(paths)
    assert not any(
        path.startswith(checker.FORBIDDEN_RUNNER_PREFIX) for path in paths
    )


def test_closure_contains_only_safe_python_source_paths() -> None:
    for relative in checker.discover_source_closure():
        checker._safe_source_path(relative)
        path = Path(relative)
        folded = tuple(part.casefold() for part in path.parts)
        assert path.suffix == ".py"
        assert not set(folded).intersection(checker.FORBIDDEN_PATH_PARTS)
        assert not any(part.startswith("sealed") for part in folded)
        assert not any(part.startswith("heldout") for part in folded)
        assert not any(part.startswith("held_out") for part in folded)


def test_discovery_honors_ignores_and_source_reads_reject_symlinks(
    tmp_path: Path,
) -> None:
    source = CHECKER_PATH.read_text(encoding="utf-8")
    base_source = (ROOT / checker.BASE_CHECKER_PATH).read_text(encoding="utf-8")
    assert ".rglob(" not in base_source
    assert '"rg",\n            "--files"' in base_source
    assert '("lewm_worlds", ROOT / "lewm_worlds/lewm_worlds")' in source
    assert '"lewm_worlds/lewm_worlds"' in source
    assert "--no-ignore" not in base_source
    assert '"-u"' not in base_source
    for exclusion in (
        "!**/sealed/**",
        "!**/sealed_*/**",
        "!**/.generated/**",
    ):
        assert exclusion in base_source
    assert "heldout_" in source
    assert "held_out_" in source
    assert "os.O_NOFOLLOW" in base_source

    target = tmp_path / "source.py"
    target.write_text("value = 1\n", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    with pytest.raises((OSError, PermissionError)):
        checker._read_regular_source(link)


def test_discovery_reads_only_returned_safe_source_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed: list[str] = []
    original_read = checker._BASE._read_regular_source

    def recording_read(path: Path) -> bytes:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        checker._safe_source_path(relative)
        accessed.append(relative)
        return original_read(path)

    monkeypatch.setattr(checker._BASE, "_read_regular_source", recording_read)
    paths = set(checker.discover_source_closure())
    assert set(accessed) == paths
    assert "lewm_worlds/lewm_worlds/manifest.py" in accessed
    assert not any("sealed" in Path(path).parts for path in accessed)
    assert not any("heldout" in Path(path).parts for path in accessed)
    assert not any("held_out" in Path(path).parts for path in accessed)


def test_integrity_replacement_uses_a_fresh_manifest_without_schema_drift() -> None:
    assert checker.MANIFEST_PATH.relative_to(ROOT).as_posix() == (
        "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
        "integrity_replacement_v3_source_manifest_2026-07-29.json"
    )
    assert checker.SCHEMA == (
        "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_source_manifest"
    )


def test_cli_exposes_emit_write_verify_and_tracked_modes() -> None:
    source = CHECKER_PATH.read_text(encoding="utf-8")
    assert '"--emit"' in source
    assert '"--write"' in source
    assert '"--require-tracked"' in source
    assert "verify_manifest(require_tracked=args.require_tracked)" in source
