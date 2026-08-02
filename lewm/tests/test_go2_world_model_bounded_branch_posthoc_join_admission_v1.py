from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as consumer
from scripts import materialize_go2_world_model_bounded_branch_posthoc_join_admission_v1 as admission


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    REPO_ROOT
    / ".generated/dev/lewm-go2-wm-bounded-branch-experiment-integrity-replacement-v1"
)
PLAN = (
    REPO_ROOT
    / "docs/lewm_go2_world_model_bounded_branch_integrity_replacement_v1_"
    "exact_plan_2026-08-02.json"
)
CALIBRATION = (
    REPO_ROOT
    / ".generated/dev/lewm-go2-wm-counterfactual-calibration-v3-textured-v03-"
    "posthoc-analysis-v1/calibration_receipt.json"
)


def _bindings() -> dict[str, dict[str, object]]:
    return {
        "consumed_terminal": admission._file_binding(
            SOURCE_ROOT / "terminal_supervision.json"
        ),
        "physics_result": admission._file_binding(
            SOURCE_ROOT / "physics_result.json"
        ),
        "physics_receipt_check": admission._file_binding(
            SOURCE_ROOT / "physics_receipt_check.json"
        ),
        "collection_plan": admission._file_binding(PLAN),
        "calibration_receipt": admission._file_binding(CALIBRATION),
    }


@pytest.fixture(scope="module")
def real_derivation():
    if not SOURCE_ROOT.is_dir():
        pytest.skip("bounded posthoc source corpus is unavailable")
    bindings = _bindings()
    before = admission._source_inventory(SOURCE_ROOT)
    derived = admission.derive_documents_v1(
        terminal_binding=bindings["consumed_terminal"],
        physics_binding=bindings["physics_result"],
        physics_check_binding=bindings["physics_receipt_check"],
        plan_binding=bindings["collection_plan"],
        calibration_receipt_binding=bindings["calibration_receipt"],
        verify_textured_pixels=False,
    )
    assert admission._source_inventory(SOURCE_ROOT) == before
    return bindings, before, derived


def test_real_derivation_matches_registered_metadata(real_derivation) -> None:
    _bindings_value, _inventory, derived = real_derivation
    assert len(derived.rgb_manifest["artifacts"]) == 3_072
    assert len(derived.rows["train"]) == 128
    assert len(derived.rows["eval"]) == 128
    for name, expected in admission.EXPECTED_LEAVES.items():
        raw = derived.raw_by_leaf[name]
        assert len(raw) == expected["byte_count"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]


def test_binding_or_root_drift_fails_closed(real_derivation) -> None:
    bindings, _inventory, _derived = real_derivation
    changed = dict(bindings["physics_result"])
    changed["file_sha256"] = "0" * 64
    with pytest.raises(Exception):
        admission.derive_documents_v1(
            terminal_binding=bindings["consumed_terminal"],
            physics_binding=changed,
            physics_check_binding=bindings["physics_receipt_check"],
            plan_binding=bindings["collection_plan"],
            calibration_receipt_binding=bindings["calibration_receipt"],
            verify_textured_pixels=False,
        )


def test_split_root_loader_reconstructs_normal_bundle(
    real_derivation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings, inventory, derived = real_derivation
    derived_root = tmp_path / "posthoc"
    derived_root.mkdir()
    monkeypatch.setattr(admission, "DEFAULT_OUTPUT_ROOT", derived_root)
    leaves = {}
    for name, expected in admission.EXPECTED_LEAVES.items():
        selected = derived_root / str(expected["name"])
        admission._write_exclusive(selected, derived.raw_by_leaf[name])
        leaves[name] = admission._file_binding(selected)
    manifest = {
        "schema": admission.MANIFEST_SCHEMA,
        "status": admission.MANIFEST_STATUS,
        "citable_as_scientific_evidence": False,
        "original_attempt_completed_successfully": False,
        "authorizes_retry_or_resume": False,
        "source_receipt_root": str(SOURCE_ROOT.resolve()),
        "derived_output_root": str(derived_root.resolve()),
        "input_bindings": bindings,
        "source_inventory_before": inventory,
        "source_inventory_after": inventory,
        "counts": admission.EXPECTED_COUNTS,
        "rgb_artifacts": 3_072,
        "role_scene_counts": {"train": 16, "eval": 16},
        "derived_leaf_bindings": leaves,
        "consumer_compatibility_projection": (
            admission.CONSUMER_COMPATIBILITY_PROJECTION
        ),
    }
    manifest_path = derived_root / "manifest.json"
    admission._write_json_exclusive(manifest_path, manifest)
    manifest_binding = admission._file_binding(manifest_path)
    bundle = admission.load_posthoc_bundle_v1(
        derived_root,
        expected_manifest_byte_count=int(manifest_binding["byte_count"]),
        expected_manifest_sha256=str(manifest_binding["file_sha256"]),
    )
    assert bundle.root == SOURCE_ROOT.resolve()
    assert len(bundle.groups_by_role["train"]) == 128
    assert len(bundle.groups_by_role["eval"]) == 128
    assert len(bundle.artifacts) == 3_072
    assert bundle.access_audit["rgb_leaf_open_count"] == 0
    first_artifact = sorted(bundle.artifacts)[0]
    assert consumer.read_bound_rgb_bytes_v1(bundle, first_artifact)
    assert admission._source_inventory(SOURCE_ROOT) == inventory


def test_source_has_no_generation_entrypoint_or_legacy_join_call() -> None:
    source_path = Path(admission.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    assert not any(
        token in name
        for name in imported
        for token in ("collect", "render_replay", "genesis")
    )
    assert "join_pilot" not in calls


def test_consumed_terminal_is_not_relabelled_success(real_derivation) -> None:
    bindings, _inventory, _derived = real_derivation
    terminal = json.loads(Path(str(bindings["consumed_terminal"]["path"])).read_text())
    assert terminal["status"] == "CONSUMED_TERMINAL_FAILURE"
    assert terminal["joined_manifest_binding"] is None
    assert terminal["joined_receipt_check_binding"] is None
    assert admission.TERMINAL_SUCCESS != terminal["status"]
