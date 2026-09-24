from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "scripts/check_go2_multires_probe_v3_science_identity.py"
SPEC = importlib.util.spec_from_file_location(
    "_test_go2_multires_probe_v3_science_identity_checker",
    CHECKER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_complete_source_only_identity_guard_reads_only_exact_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    original = checker._read_regular_source

    def audited_read(root: Path, relative: str) -> bytes:
        assert relative in checker.READABLE_SOURCE_PATHS
        parts = Path(relative).parts
        assert ".generated" not in parts
        assert "checkpoints" not in parts
        assert not any(
            part == "sealed" or part.startswith("sealed_")
            for part in parts
        )
        observed.append(relative)
        return original(root, relative)

    monkeypatch.setattr(checker, "_read_regular_source", audited_read)
    report = checker.verify_all()
    assert report["generated_runtime_checkpoint_or_gpu_open_count"] == 0
    assert report["science_identity"] == {
        "deep_equal_source_count": 3,
        "science_contract_sha256": checker.SCIENCE_SHA256,
    }
    assert len(report["v2_frozen_sources"]) == 12
    assert set(observed).issubset(checker.READABLE_SOURCE_PATHS)
    assert checker.V2_SCHEDULE_ADAPTER in observed
    assert checker.COMPAT_CONTRACT in observed
    assert checker.COMPAT_RUNNER in observed
    assert checker.COMPAT_LAUNCHER in observed


def test_exact_twelve_v2_source_bindings_are_frozen() -> None:
    assert checker.V2_FROZEN_SOURCE_SHA256 == {
        checker.V2_CONTRACT:
            "53e045a208a39705e12537a698c20d6d1c4508cc13145ebdb04cd66f494ad1fd",
        checker.V2_RUNNER:
            "5fdec79263e904b41b279eb1560b60ab2f9a89384fd032b31330d68b9d003c45",
        checker.V2_LAUNCHER:
            "d721334113a9c580dc2db4a3444c80ab3f9b08d268b56090a95236f33a947296",
        checker.V2_TEST:
            "a49050ffe3f46ff12c6901894fede47c4e5159c84f06b66fc8dce6ae75d8000c",
        checker.V2_CLOSURE_CHECKER:
            "c5010ba4dec12c1d23d1c158ccdd35f20c0dc6e3fab0b39916912f2790866b79",
        checker.V2_CLOSURE_TEST:
            "720f6c42f41bc350a0854c7276a875499c7516b705fcc179f535a690fa66a431",
        checker.V2_IDENTITY_CHECKER:
            "e87402f57abffa70340161fc54c2285d624747933d5a12d4fbed1b4422acab6e",
        checker.V2_IDENTITY_TEST:
            "5211f7bdd77a018a42ad920aa47ebfc9ac63c0b0036665e5e93c80489a5792d8",
        checker.V2_SCHEDULE_ADAPTER:
            "a8efe19da92c9c2107f11be38db8ed80e66aedca3ef41af0428ab13d50f56bd1",
        checker.V2_SCHEDULE_TEST:
            "340828cb55a03da575ccfb8242ff3e3db8b8f15527d43891b737cfad8a5b2204",
        checker.MODEL_PATH: checker.MODEL_SHA256,
        checker.MODEL_TEST:
            "a241910c83bc44cf15b56270659becf1def66f358f3f2bb1a89d89a9bce30fae",
    }
    assert checker.verify_v2_frozen_sources() == (
        checker.V2_FROZEN_SOURCE_SHA256
    )


def test_science_contract_evaluator_rejects_filesystem_calls() -> None:
    raw = (ROOT / checker.V2_CONTRACT).read_text(encoding="utf-8")
    tree = ast.parse(raw)
    science = checker._science_function(tree, "science_contract")
    science.body.insert(
        0,
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id="open", ctx=ast.Load()),
                args=[ast.Constant(".generated/forbidden")],
                keywords=[],
            )
        ),
    )
    with pytest.raises(checker.GuardFailure, match="unauthorized call"):
        checker._assert_science_ast_is_pure(science)


def test_definition_delta_rejects_an_unrelated_fourth_change() -> None:
    before = ast.parse(
        "def science(x):\n"
        "    return x + 1\n"
        "\n"
        "def lifecycle(x):\n"
        "    return x\n"
    )
    after = ast.parse(
        "def science(x):\n"
        "    return x + 2\n"
        "\n"
        "def lifecycle(x):\n"
        "    return x\n"
        "\n"
        "def unrelated_tuning():\n"
        "    return 3\n"
    )
    added, removed, changed = checker._definition_delta(before, after)
    assert added == {"unrelated_tuning"}
    assert removed == set()
    assert changed == {"science"}


def test_authority_bindings_and_science_hash_are_canonical() -> None:
    report = checker.verify_preregistration_authority()
    assert report["decision_file_sha256"] == (
        "94ab2ca50cdc5c33008a411aafc07461684d8564433a9fd787f68308db04b6a2"
    )
    assert report["preregistration_content_sha256"] == (
        "64da13d6e38a8c1ee2a1bc87b9917611097023a36939ee4305be9a4e85f602b7"
    )
    assert report["preregistration_review_content_sha256"] == (
        "ca42b49c9360946dd5ab5ad29e488a7354ea55f788bc566f528520256bf8aa23"
    )
    science = checker.verify_science_identity()
    assert science["science_contract_sha256"] == checker.SCIENCE_SHA256
    assert len(json.dumps(science, allow_nan=False)) > 0


def test_model_runtime_and_output_roots_are_exactly_distinct() -> None:
    report = checker.verify_model_and_roots()
    assert report["model_file_sha256"] == checker.MODEL_SHA256
    assert report["model_family"] == checker.MODEL_FAMILY
    assert report["model_runtime_version"] == checker.MODEL_RUNTIME_VERSION
    assert report["v2_output_root"] == checker.V2_ROOT
    assert report["v3_output_root"] == checker.V3_ROOT
    assert report["v2_output_root"] != report["v3_output_root"]
    assert report["unchanged_common_literal_binding_count"] >= 40


def test_schedule_adapter_is_frozen_pure_two_stage_adapter() -> None:
    report = checker.verify_schedule_adapter()
    assert report["adapter_file_sha256"] == checker.V2_SCHEDULE_ADAPTER_SHA256
    assert report["functions"] == [
        "finalize_train_identity",
        "validate_bound_schedule_phase_a",
    ]
    assert report["normalized_schedule_content_sha256"] == (
        "893c48b2c2c591dbc90469e5a19a74e70bd54f96689b63881c216605255c0e5d"
    )


def test_receipt_and_pre_ledger_mechanisms_are_ast_guarded() -> None:
    assert checker.verify_operational_mechanisms() == {
        "canonical_to_dict_precedes_dataclass_normalization": True,
        "failure_receipt_direct_reservation_binding": True,
        "fsynced_open_attempt_and_outcome_ledger": True,
        "post_reservation_pre_ledger_terminalization": True,
        "prior_v1_and_v2_runtime_output_open_authorized": False,
        "schedule_validation_precedes_n320_and_raw": True,
    }


def test_compatibility_checker_is_exactly_synthetic_and_strict() -> None:
    assert checker.verify_compatibility_checker_static() == {
        "child_order": ["grid_sample", "scatter_add"],
        "compatibility_output_root": (
            ".generated/"
            "go2_rgb_multiresolution_perception_r9700_strict_compatibility_v1"
        ),
        "generated_dataset_checkpoint_model_open_count": 0,
        "grid_sample_call_count": 20,
        "scatter_add_call_count": 148,
        "strict_warn_only": False,
        "synthetic_only": True,
        "v3_probe_root_inspection_or_reservation_authorized": False,
    }


def test_delta_surface_is_exactly_three_authorized_mechanisms() -> None:
    report = checker.verify_delta_surface()
    assert report["authorized_operational_delta_ids"] == [
        "canonical_initialization_receipt_normalization",
        "post_reservation_pre_ledger_terminalization",
        "synthetic_r9700_strict_determinism_compatibility_checker",
    ]
    assert report["science_bearing_contract_definition_count"] == 7
    assert report["science_bearing_runner_definition_count"] == 12
    assert set(report["paired_source_deltas"]) == set(checker.PAIRED_SOURCES)


def test_generated_paths_are_rejected_before_any_source_open() -> None:
    with pytest.raises(PermissionError, match="outside the source-only allowlist"):
        checker._safe_relative_source(
            ".generated/go2_shared_observable_camera_ray_jepa_v5/"
            "rgb_multiresolution_perception_probe_v3/reservation.json"
        )
