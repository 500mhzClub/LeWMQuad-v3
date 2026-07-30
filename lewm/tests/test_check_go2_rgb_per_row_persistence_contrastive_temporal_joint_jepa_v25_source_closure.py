from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / (
    "scripts/check_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v25_source_closure.py"
)


def _load(name: str):
    for module_name in tuple(sys.modules):
        if module_name.startswith("_lewm_v25_per_row_temporal_source_closure_base"):
            sys.modules.pop(module_name, None)
        elif "source_closure_base" in module_name and module_name.startswith("_lewm_v24"):
            sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_checker_is_denied_and_binds_frozen_v24(capsys) -> None:
    checker = _load("_v25_checker_denial")
    receipt = checker.execution_denial_receipt_v25()
    assert receipt["execution_authorized"] is False
    assert receipt["recovery_state_opened"] is False
    assert checker.BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT == (
        "2b6178a4d876dc17c45fb340a4ab03ee302649b0"
    )
    assert checker.PREREGISTRATION_COMMIT == (
        "f00e20df3b429f9242516ac38f67fea587e04b22"
    )
    assert checker.V24_SCIENTIFIC_RESULT_COMMIT == (
        "2824c80c54fc7502b1413b3371fc87c9206f82a2"
    )
    assert checker.main(["--emit"]) == 0
    assert checker.SCHEMA in capsys.readouterr().out


def test_preregistration_and_v24_result_bindings_are_exact() -> None:
    checker = _load("_v25_checker_bindings")
    for relative, expected_sha, expected_bytes in (
        (
            checker.PREREGISTRATION_RELATIVE_PATH,
            checker.PREREGISTRATION_FILE_SHA256,
            checker.PREREGISTRATION_BYTE_COUNT,
        ),
        (
            checker.V24_SCIENTIFIC_RESULT_RELATIVE_PATH,
            checker.V24_SCIENTIFIC_RESULT_FILE_SHA256,
            checker.V24_SCIENTIFIC_RESULT_BYTE_COUNT,
        ),
    ):
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == expected_bytes
        assert hashlib.sha256(raw).hexdigest() == expected_sha


def test_closure_adds_only_v25_entrypoints_and_forces_v24_parents() -> None:
    checker = _load("_v25_checker_entrypoints")
    assert checker.ENTRYPOINTS[1:] == checker.IMPLEMENTATION_PATHS
    assert set(checker.V24_PARENT_ENTRYPOINTS).issubset(
        checker.FORCED_DYNAMIC_SOURCES
    )
    manifest = checker.build_manifest()
    paths = tuple(manifest["source_paths"])
    assert all(path in paths for path in checker.IMPLEMENTATION_PATHS)
    # Recursive runtime closure excludes checker tooling; the predecessor
    # checker remains a separately bound clean-export/review extra.
    assert checker.BASE_CHECKER_PATH not in paths
    assert checker.PREREGISTRATION_RELATIVE_PATH not in paths
    assert checker.V24_SCIENTIFIC_RESULT_RELATIVE_PATH not in paths


def test_candidate_contains_no_runtime_recovery_or_protected_material() -> None:
    checker = _load("_v25_checker_custody")
    manifest = checker.build_manifest()
    paths = tuple(manifest["source_paths"])
    assert len(paths) == len(set(paths))
    assert all((ROOT / path).is_file() and not (ROOT / path).is_symlink() for path in paths)
    forbidden_parts = {".generated", "sealed", "heldout", "held_out"}
    assert all(not (forbidden_parts & set(Path(path).parts)) for path in paths)
    assert not any("recovery/update_400_training_state" in path for path in paths)
    assert manifest["excluded_runtime_categories"] == list(
        checker.EXCLUDED_RUNTIME_CATEGORIES
    )
