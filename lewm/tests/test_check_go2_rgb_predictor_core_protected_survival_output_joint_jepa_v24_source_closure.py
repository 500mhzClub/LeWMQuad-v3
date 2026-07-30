from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / (
    "scripts/check_go2_rgb_predictor_core_protected_survival_output_joint_"
    "jepa_v24_source_closure.py"
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_IMPORTED_BEFORE = set(sys.modules)
checker = _load("_test_v24_core_protected_source_closure", CHECKER_PATH)
_IMPORTED_BY_CHECKER = set(sys.modules) - _IMPORTED_BEFORE


def test_checker_import_is_source_only_denied_and_uses_v24_identity() -> None:
    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in _IMPORTED_BY_CHECKER
        for prefix in ("torch", "numpy", "PIL", "lewm", "scripts")
    )
    assert checker.SCHEMA == (
        "lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
        "v24_source_manifest"
    )
    deepest = checker._V23._V22._V21._V20._V18._V13._BASE
    assert deepest.SCHEMA == checker.SCHEMA
    assert checker.MANIFEST_PATH.relative_to(ROOT).as_posix() == (
        "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_"
        "v24_source_manifest_2026-07-30.json"
    )
    assert checker.EXECUTION_AUTHORIZED is False
    denial = checker.execution_denial_receipt_v24()
    assert denial["status"] == "DENIED_INCOMPLETE_SOURCE_LIFECYCLE"
    assert denial["execution_authorized"] is False
    assert denial["checkpoint_opened"] is False


def test_frozen_preregistration_and_v23_terminal_result_are_exactly_bound() -> None:
    expected = (
        (
            checker.PREREGISTRATION_RELATIVE_PATH,
            "475f1867149f5c5b764973bb5a371de83c29c3eb",
            "ad0514668b20fd3bb58a2c70e71bb153428f3a9b121c1f8b64ca6e08965c6933",
            12_137,
        ),
        (
            checker.V23_SCIENTIFIC_RESULT_RELATIVE_PATH,
            "04b0fa48c6c4e10868c2f302bc51100394e3907e",
            "753c91babd4f7116444654167d2507ffb52d22f970fc926c05d287683954c994",
            20_640,
        ),
    )
    assert checker.PREREGISTRATION_COMMIT == expected[0][1]
    assert checker.V23_SCIENTIFIC_RESULT_COMMIT == expected[1][1]
    assert checker.V23_SCIENTIFIC_RESULT_CONTENT_SHA256 == (
        "a5a6b8aa7312706d2ae3a5b53e39370462e9de6eda6b7a2ca4e2e0226a518ed8"
    )
    for relative, _, digest, byte_count in expected:
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == byte_count
        assert hashlib.sha256(raw).hexdigest() == digest
    predecessor = json.loads(
        (ROOT / checker.V23_SCIENTIFIC_RESULT_RELATIVE_PATH).read_text()
    )
    assert (
        predecessor["content_sha256"]
        == checker.V23_SCIENTIFIC_RESULT_CONTENT_SHA256
    )


def test_entrypoints_bind_exact_v24_implementation_and_retain_v23_parents() -> None:
    assert checker.ENTRYPOINTS == (
        (
            "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_"
            "v18_object_space_height_volume.py"
        ),
        checker.RUNNER_RELATIVE_PATH,
        checker.EXECUTOR_RELATIVE_PATH,
        checker.LAUNCHER_RELATIVE_PATH,
    )
    expected_parents = {
        (
            "scripts/run_go2_rgb_action_prior_residualized_wrong_scene_survival_"
            "output_joint_jepa_v23.py"
        ),
        (
            "scripts/execute_go2_rgb_action_prior_residualized_wrong_scene_"
            "survival_output_joint_jepa_v23.py"
        ),
        (
            "scripts/launch_go2_rgb_action_prior_residualized_wrong_scene_"
            "survival_output_joint_jepa_v23.py"
        ),
    }
    assert set(checker.V23_PARENT_ENTRYPOINTS) == expected_parents
    assert expected_parents.issubset(set(checker.FORCED_DYNAMIC_SOURCES))


def test_lifecycle_paths_are_exact_future_source_records_not_runtime_inputs() -> None:
    assert checker.LIFECYCLE_PATHS == {
        "preregistration": checker.PREREGISTRATION_RELATIVE_PATH,
        "predecessor_scientific_result": (
            checker.V23_SCIENTIFIC_RESULT_RELATIVE_PATH
        ),
        "source_manifest": checker.MANIFEST_RELATIVE_PATH,
        "source_review": (
            "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_"
            "jepa_v24_source_review_2026-07-30.json"
        ),
        "clean_export_certification": (
            "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_"
            "jepa_v24_clean_export_certification_2026-07-30.json"
        ),
        "execution_authority": (
            "docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_"
            "jepa_v24_execution_authorization_2026-07-30.json"
        ),
    }
    assert not any(
        "checkpoint" in relative.casefold()
        for relative in checker.LIFECYCLE_PATHS.values()
    )


def test_candidate_is_complete_exact_bound_and_contains_no_runtime_material() -> None:
    manifest = checker.build_manifest()
    assert manifest["schema"] == checker.SCHEMA
    assert manifest["source_count"] == len(manifest["source_paths"])
    assert manifest["source_count"] == len(manifest["source_bindings"])
    assert len(manifest["source_paths"]) == len(set(manifest["source_paths"]))
    assert manifest["source_paths"] == [
        binding["path"] for binding in manifest["source_bindings"]
    ]
    assert set(checker.ENTRYPOINTS).issubset(set(manifest["source_paths"]))
    assert set(checker.V23_PARENT_ENTRYPOINTS).issubset(
        set(manifest["source_paths"])
    )
    bindings = {row["path"]: row for row in manifest["source_bindings"]}
    for relative in checker.IMPLEMENTATION_PATHS:
        raw = (ROOT / relative).read_bytes()
        assert bindings[relative] == {
            "path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
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
        assert not any(
            part.startswith(("sealed", "heldout", "held_out"))
            for part in folded
        )
    source = CHECKER_PATH.read_text(encoding="utf-8")
    assert 'mode.add_argument("--emit"' in source
    assert 'mode.add_argument("--write"' in source
    assert 'parser.add_argument("--require-tracked"' in source
    assert "--no-ignore" not in source
