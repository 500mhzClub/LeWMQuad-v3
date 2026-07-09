from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from lewm_worlds.manifest import manifest_sha256
from lewm_worlds.families import build_family_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/materialize_go2_generalization_scenes.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("materialize_generalization", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_development_scene_reconstructs_with_exact_hash() -> None:
    module = _load_script()
    payload = json.loads(
        (REPO_ROOT / "config/go2_generalization_v3/development.json").read_text()
    )
    record = payload["validation_scenes"][0]

    manifest = module.manifest_from_record(record)

    assert manifest.scene_id == record["scene_id"]
    assert manifest_sha256(manifest) == record["manifest_sha256"]


def test_deployment_family_record_reconstructs_with_exact_hash() -> None:
    module = _load_script()
    expected = build_family_manifest(
        scene_seed=2026070906,
        family="go2_deployment_medium_maze",
        split="train",
        difficulty_tier=None,
    )
    record = {
        "topology_seed": expected.topology_seed,
        "family": expected.family,
        "source_split": expected.split,
        "scene_id": expected.scene_id,
        "manifest_sha256": manifest_sha256(expected),
    }

    actual = module.manifest_from_record(record)

    assert actual.scene_id == expected.scene_id
    assert manifest_sha256(actual) == manifest_sha256(expected)


def test_materializer_rejects_sealed_paths_before_opening() -> None:
    module = _load_script()

    try:
        module._development_path_guard(
            Path("config/go2_generalization_v4/sealed_test.json"),
            label="development manifest",
        )
    except ValueError as error:
        assert "development-only" in str(error)
    else:
        raise AssertionError("sealed materialization path was accepted")
