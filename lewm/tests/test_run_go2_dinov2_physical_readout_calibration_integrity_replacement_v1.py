from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import (
    run_go2_dinov2_physical_readout_calibration_integrity_replacement_v1
    as runner,
)


def _stored_document() -> dict[str, object]:
    return {
        "schema": runner.compatibility.TASK_RELEVANCE_SCHEMA,
        "status": runner.compatibility.TASK_RELEVANCE_PASS_STATUS,
        "thresholds": {
            "minimum_reference_candidate_rgb_ssim": 0.99,
            "required_paired_nearest_neighbour_retrieval_count": 32,
        },
        "measurements": {
            "pixels": {
                "minimum_reference_candidate_rgb_ssim": 0.999873849744854,
            },
            "frozen_predecessor_descriptor_retrieval": {
                "maximum_paired_descriptor_distance": 0.0014817728354341111,
                "paired_nearest_neighbour_retrieval_count": 32,
            },
        },
        "bindings": {
            "source": {
                "path": "/development/source.json",
                "file_sha256": "a" * 64,
                "byte_count": 1,
            }
        },
    }


def _authority(output_root: Path) -> dict[str, object]:
    return {
        "output_root": str(output_root),
        "preregistration_binding": {
            "path": "/development/preregistration",
            "sha256": "0" * 64,
            "byte_count": 1,
        },
        "input_bindings": {
            "original_terminal_failure_review": {
                "path": "/development/failure-review",
                "sha256": "1" * 64,
                "byte_count": 1,
            },
            "stored_task_relevance_result": {
                "path": "/development/result",
                "sha256": "2" * 64,
                "byte_count": 1,
            },
            "stored_task_relevance_review": {
                "path": "/development/review",
                "sha256": "3" * 64,
                "byte_count": 1,
            },
        },
        "source_bindings": {
            "task_relevance_evaluator": {
                "path": "/development/evaluator.py",
                "sha256": "4" * 64,
                "byte_count": 1,
            }
        },
        "environment": {
            "python": "/usr/bin/python3.12",
            "torch": "synthetic",
            "hip": "synthetic",
            "numpy": "synthetic",
            "pillow": "synthetic",
        },
    }


def _authority_binding() -> dict[str, object]:
    return {
        "path": "/development/authority",
        "sha256": "5" * 64,
        "byte_count": 1,
    }


def _install_live_path(monkeypatch, recomputed: dict[str, object], calls: list[str]):
    def live_evaluator(*_args, **_kwargs):
        calls.append("live_evaluator")
        return recomputed

    def strict_loader():
        calls.append("strict_loader_enter")
        runner.task_relevance.evaluate_task_relevance_v1(synthetic=True)
        calls.append("strict_loader_return")
        return SimpleNamespace(bundle=True)

    monkeypatch.setattr(
        runner.task_relevance, "evaluate_task_relevance_v1", live_evaluator
    )
    monkeypatch.setattr(
        runner.original.screen_data,
        "load_bound_posthoc_bundle_v1",
        strict_loader,
    )
    return live_evaluator, strict_loader


def test_scoped_admission_publishes_receipt_before_strict_loader_returns_and_restores(
    tmp_path, monkeypatch
) -> None:
    stored = _stored_document()
    recomputed = deepcopy(stored)
    recomputed["measurements"]["pixels"][
        "minimum_reference_candidate_rgb_ssim"
    ] = 0.9998738497448542
    calls: list[str] = []
    live_evaluator, strict_loader = _install_live_path(
        monkeypatch, recomputed, calls
    )
    monkeypatch.setattr(
        runner, "_load_stored_task_relevance_v1", lambda _authority: stored
    )
    authority = _authority(tmp_path)

    with runner.scoped_compatibility_admission_v1(
        authority,
        authority_binding=_authority_binding(),
    ) as state:
        bundle = runner.original.screen_data.load_bound_posthoc_bundle_v1()
        assert bundle.bundle is True
        assert state["evaluator_calls"] == 1
        assert state["loader_calls"] == 1
        assert state["receipt_binding"] is not None
        assert (tmp_path / "compatibility_receipt.json").is_file()
        receipt = runner.json.loads(
            (tmp_path / "compatibility_receipt.json").read_text()
        )
        assert receipt["status"] == runner.COMPATIBILITY_RECEIPT_STATUS
        assert receipt["admission"]["differing_paths"] == [
            runner.compatibility.SSIM_DOTTED_PATH
        ]
        assert calls == [
            "strict_loader_enter",
            "live_evaluator",
            "strict_loader_return",
        ]

    assert runner.task_relevance.evaluate_task_relevance_v1 is live_evaluator
    assert (
        runner.original.screen_data.load_bound_posthoc_bundle_v1
        is strict_loader
    )


def test_scoped_admission_rejects_second_field_and_restores_without_receipt(
    tmp_path, monkeypatch
) -> None:
    stored = _stored_document()
    recomputed = deepcopy(stored)
    recomputed["measurements"]["pixels"][
        "minimum_reference_candidate_rgb_ssim"
    ] = 0.9998738497448542
    recomputed["measurements"]["frozen_predecessor_descriptor_retrieval"][
        "maximum_paired_descriptor_distance"
    ] = 0.0014817737103522426
    calls: list[str] = []
    live_evaluator, strict_loader = _install_live_path(
        monkeypatch, recomputed, calls
    )
    monkeypatch.setattr(
        runner, "_load_stored_task_relevance_v1", lambda _authority: stored
    )

    with pytest.raises(
        runner.compatibility.CompatibilityAdmissionError,
        match="outside the singleton SSIM path",
    ):
        with runner.scoped_compatibility_admission_v1(
            _authority(tmp_path),
            authority_binding=_authority_binding(),
        ):
            runner.original.screen_data.load_bound_posthoc_bundle_v1()

    assert not (tmp_path / "compatibility_receipt.json").exists()
    assert runner.task_relevance.evaluate_task_relevance_v1 is live_evaluator
    assert (
        runner.original.screen_data.load_bound_posthoc_bundle_v1
        is strict_loader
    )


def test_receipt_publication_failure_stops_loader_before_return_and_restores(
    tmp_path, monkeypatch
) -> None:
    stored = _stored_document()
    recomputed = deepcopy(stored)
    calls: list[str] = []
    live_evaluator, strict_loader = _install_live_path(
        monkeypatch, recomputed, calls
    )
    monkeypatch.setattr(
        runner, "_load_stored_task_relevance_v1", lambda _authority: stored
    )
    monkeypatch.setattr(
        runner.original,
        "_write_json_exclusive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        with runner.scoped_compatibility_admission_v1(
            _authority(tmp_path),
            authority_binding=_authority_binding(),
        ):
            runner.original.screen_data.load_bound_posthoc_bundle_v1()

    assert calls == ["strict_loader_enter", "live_evaluator"]
    assert runner.task_relevance.evaluate_task_relevance_v1 is live_evaluator
    assert (
        runner.original.screen_data.load_bound_posthoc_bundle_v1
        is strict_loader
    )


def test_loader_cannot_return_without_evaluator_and_receipt(tmp_path, monkeypatch) -> None:
    stored = _stored_document()
    live_evaluator = lambda *_args, **_kwargs: deepcopy(stored)
    bypass_loader = lambda: SimpleNamespace(bundle=True)
    monkeypatch.setattr(
        runner.task_relevance, "evaluate_task_relevance_v1", live_evaluator
    )
    monkeypatch.setattr(
        runner.original.screen_data,
        "load_bound_posthoc_bundle_v1",
        bypass_loader,
    )
    monkeypatch.setattr(
        runner, "_load_stored_task_relevance_v1", lambda _authority: stored
    )

    with pytest.raises(
        runner.IntegrityReplacementRunnerError,
        match="not published before loader return",
    ):
        with runner.scoped_compatibility_admission_v1(
            _authority(tmp_path), authority_binding=_authority_binding()
        ):
            runner.original.screen_data.load_bound_posthoc_bundle_v1()

    assert not (tmp_path / "compatibility_receipt.json").exists()
    assert runner.task_relevance.evaluate_task_relevance_v1 is live_evaluator
    assert (
        runner.original.screen_data.load_bound_posthoc_bundle_v1
        is bypass_loader
    )


@pytest.mark.parametrize(
    "status",
    [runner.original.PASS_STATUS, runner.original.STOP_STATUS, runner.original.FAIL_STATUS],
)
def test_execute_preserves_original_scientific_terminal_status(
    tmp_path, monkeypatch, status: str
) -> None:
    output_root = tmp_path / "replacement"
    output_root.mkdir()
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(runner, "ORIGINAL_OUTPUT_ROOT", tmp_path / "original")
    stored = _stored_document()
    recomputed = deepcopy(stored)
    recomputed["measurements"]["pixels"][
        "minimum_reference_candidate_rgb_ssim"
    ] = 0.9998738497448542
    calls: list[str] = []
    _install_live_path(monkeypatch, recomputed, calls)
    monkeypatch.setattr(
        runner, "_load_stored_task_relevance_v1", lambda _authority: stored
    )

    def original_execute(authority, *, authority_binding):
        bundle = runner.original.screen_data.load_bound_posthoc_bundle_v1()
        assert bundle.bundle is True
        return {"status": status, "authority_binding": authority_binding}

    monkeypatch.setattr(runner.original, "execute_v1", original_execute)
    report = runner.execute_replacement_v1(
        _authority(output_root), authority_binding=_authority_binding()
    )
    assert report["status"] == status
    assert (output_root / "compatibility_receipt.json").is_file()


def test_output_root_is_fresh_replacement_only(monkeypatch, tmp_path) -> None:
    replacement = tmp_path / "replacement"
    original = tmp_path / "original"
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT_ROOT", replacement)
    monkeypatch.setattr(runner, "ORIGINAL_OUTPUT_ROOT", original)
    assert runner._validate_output_root_v1(str(replacement)) == replacement  # noqa: SLF001
    with pytest.raises(
        runner.IntegrityReplacementRunnerError, match="output root changed"
    ):
        runner._validate_output_root_v1(str(original))  # noqa: SLF001
    with pytest.raises(
        runner.IntegrityReplacementRunnerError, match="output root changed"
    ):
        runner._validate_output_root_v1(str(tmp_path / "other"))  # noqa: SLF001


def test_source_review_requires_exact_registered_checks() -> None:
    preregistration = {"path": "/prereg", "sha256": "0" * 64, "byte_count": 1}
    failure = {"path": "/failure", "sha256": "1" * 64, "byte_count": 1}
    sources = {"runner": {"path": "/runner", "sha256": "2" * 64, "byte_count": 1}}
    review = {
        "schema": runner.SOURCE_REVIEW_SCHEMA,
        "status": runner.SOURCE_REVIEW_STATUS,
        "review_date": "2026-08-03",
        "reviewer": {
            "identity": "independent",
            "independence_basis": "did not author the replacement source",
        },
        "protected_material_opened": False,
        "preregistration_binding": preregistration,
        "original_failure_review_binding": failure,
        "source_bindings": sources,
        "checks": {name: True for name in runner.SOURCE_REVIEW_CHECKS},
        "audit_history": [
            {
                "finding": "synthetic review finding",
                "resolution": "FIXED_BEFORE_FREEZE",
                "evidence": "synthetic evidence",
            }
        ],
        "findings": [],
    }
    runner._validate_source_review_v1(  # noqa: SLF001
        review,
        preregistration_binding=preregistration,
        original_failure_review_binding=failure,
        source_bindings=sources,
    )
    valid_review = deepcopy(review)
    review["checks"] = {"placeholder": True}
    with pytest.raises(
        runner.IntegrityReplacementRunnerError, match="did not pass exactly"
    ):
        runner._validate_source_review_v1(  # noqa: SLF001
            review,
            preregistration_binding=preregistration,
            original_failure_review_binding=failure,
            source_bindings=sources,
        )
    for invalid in (
        {**valid_review, "reviewer": {"identity": "", "independence_basis": ""}},
        {**valid_review, "audit_history": []},
        {**valid_review, "review_date": ""},
    ):
        with pytest.raises(
            runner.IntegrityReplacementRunnerError, match="did not pass exactly"
        ):
            runner._validate_source_review_v1(  # noqa: SLF001
                invalid,
                preregistration_binding=preregistration,
                original_failure_review_binding=failure,
                source_bindings=sources,
            )


def test_source_closure_contains_original_and_replacement_claim_code() -> None:
    assert {
        "compatibility_module",
        "compatibility_test",
        "replacement_runner",
        "replacement_runner_test",
        "task_relevance_evaluator",
        "task_relevance_collector",
        "task_relevance_h6_dataset",
        "task_relevance_mask_benchmark",
        "task_relevance_parity_evaluator",
        "task_relevance_parity_supervisor",
        "task_relevance_probe",
        "task_relevance_probe_evaluator",
        "task_relevance_probe_model",
        "task_relevance_probe_trainer",
        "original_runner",
        "original_calibration_module",
    } <= set(runner.SOURCE_PATHS)
    assert runner.replacement_config_v1()["integrity_replacement"] == "v1"
    assert runner.replacement_config_v1()[
        "compatibility_allowed_differing_paths"
    ] == [runner.compatibility.SSIM_DOTTED_PATH]
