from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "scripts/check_go2_multires_probe_v2_science_identity.py"
SPEC = importlib.util.spec_from_file_location(
    "_test_go2_multires_probe_v2_science_identity_checker",
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
        assert not any(part == "sealed" or part.startswith("sealed_") for part in parts)
        observed.append(relative)
        return original(root, relative)

    monkeypatch.setattr(checker, "_read_regular_source", audited_read)
    report = checker.verify_all()
    assert report["generated_runtime_checkpoint_or_gpu_open_count"] == 0
    assert report["science_identity"] == {
        "deep_equal_source_count": 4,
        "science_contract_sha256": checker.SCIENCE_SHA256,
    }
    assert len(report["v1_frozen_sources"]) == 7
    assert set(observed).issubset(checker.READABLE_SOURCE_PATHS)
    assert checker.V2_SCHEDULE_ADAPTER in observed


def test_exact_seven_v1_source_bindings_are_frozen() -> None:
    assert checker.V1_FROZEN_SOURCE_SHA256 == {
        checker.V1_CONTRACT:
            "ffdeb2b6b3a03a1b1b65e2fe3961a8561717c8ced4d800c640f03710af40fa3b",
        checker.V1_RUNNER:
            "c84604df4933a04939c297fa68e765ec6c00e68d360da0c6ed8de5a56ba87e41",
        checker.V1_LAUNCHER:
            "adf97ed861c2f37960db1fbc171c91913847d2f4a98e553ea903d9371419f42e",
        checker.V1_TEST:
            "dba0954f9eed9d700bfe808b6911466cce8728cef247788fbcfe00b65798de0b",
        checker.V1_CLOSURE_CHECKER:
            "ac9fcaa9107ad43201b5082581c0743ebb46653ff8b51a6f09c33fc992142911",
        checker.V1_CLOSURE_TEST:
            "fb09c98b0f008eb863622dab1b4204535b719734eaf9293adb6eaefd3417f846",
        checker.MODEL_PATH: checker.MODEL_SHA256,
    }
    assert checker.verify_v1_frozen_sources() == checker.V1_FROZEN_SOURCE_SHA256


def test_science_contract_evaluator_rejects_filesystem_calls() -> None:
    raw = (ROOT / checker.V1_CONTRACT).read_text(encoding="utf-8")
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


def test_definition_delta_rejects_an_unrelated_third_change() -> None:
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
        "9df833efb3949744e66cb5263d341baef69241d4b2b1653d90ca9bf87f8ec1fb"
    )
    assert report["preregistration_content_sha256"] == (
        "264a4e3d52dd0ec658afce8c4bc54f86e9c18bbfb43229c14521b5f683a6514a"
    )
    assert report["preregistration_review_content_sha256"] == (
        "6abd1b01aa7e4df68b1fe05b0ff854124971d5b1f2f4eccd34aa42320987e04c"
    )
    science = checker.verify_science_identity()
    assert science["science_contract_sha256"] == checker.SCIENCE_SHA256
    assert len(json.dumps(science, allow_nan=False)) > 0


def test_model_runtime_and_output_roots_are_exactly_distinct() -> None:
    report = checker.verify_model_and_roots()
    assert report["model_file_sha256"] == checker.MODEL_SHA256
    assert report["model_family"] == checker.MODEL_FAMILY
    assert report["model_runtime_version"] == checker.MODEL_RUNTIME_VERSION
    assert report["v1_output_root"] == checker.V1_ROOT
    assert report["v2_output_root"] == checker.V2_ROOT
    assert report["v1_output_root"] != report["v2_output_root"]
    assert report["unchanged_common_literal_binding_count"] >= 35


def test_schedule_adapter_is_pure_two_stage_schema_adapter() -> None:
    report = checker.verify_schedule_adapter()
    assert report["functions"] == [
        "finalize_train_identity",
        "validate_bound_schedule_phase_a",
    ]
    assert report["normalized_schedule_content_sha256"] == (
        "893c48b2c2c591dbc90469e5a19a74e70bd54f96689b63881c216605255c0e5d"
    )


def test_failure_receipts_and_schedule_first_order_are_ast_guarded() -> None:
    assert checker.verify_operational_mechanisms() == {
        "failure_receipt_direct_reservation_binding": True,
        "fsynced_open_attempt_and_outcome_ledger": True,
        "schedule_validation_precedes_n320_and_raw": True,
        "v1_runtime_output_open_authorized": False,
    }


def test_delta_surface_is_exactly_the_two_operational_mechanisms() -> None:
    report = checker.verify_delta_surface()
    assert report["authorized_operational_delta_ids"] == [
        "bound_schedule_schema_adapter",
        "complete_failure_receipts",
    ]
    assert report["science_bearing_runner_definition_count"] == 7
