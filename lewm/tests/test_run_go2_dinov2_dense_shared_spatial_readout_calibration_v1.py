from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import pytest
import torch

from scripts import (
    run_go2_dinov2_dense_shared_spatial_readout_calibration_v1 as runner,
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
            },
            "parity_result": {
                "path": "/development/parity-result.json",
                "file_sha256": "b" * 64,
                "byte_count": 2,
            },
            "terminal_failure": {
                "path": "/development/terminal-failure.json",
                "file_sha256": "c" * 64,
                "byte_count": 3,
            },
            "progression_analysis": {
                "path": "/development/progression-analysis.json",
                "file_sha256": "d" * 64,
                "byte_count": 4,
            },
        },
    }


def _binding(path: str, character: str = "a") -> dict[str, object]:
    return {
        "path": path,
        "sha256": character * 64,
        "byte_count": 1,
    }


def _authority(output_root: Path) -> dict[str, object]:
    return {
        "output_root": str(output_root),
        "preregistration_binding": _binding("/development/preregistration", "0"),
        "input_bindings": {
            "prior_terminal_review": _binding("/development/prior-review", "1"),
            "prior_compatibility_receipt": _binding(
                "/development/prior-compatibility.json", "7"
            ),
            "stored_task_relevance_result": _binding(
                "/development/task-result", "2"
            ),
            "stored_task_relevance_review": _binding(
                "/development/task-review", "3"
            ),
            "eval_cache": _binding("/development/eval-cache", "4"),
        },
        "source_bindings": {
            "task_relevance_evaluator": _binding(
                "/development/task-evaluator.py", "5"
            ),
            "dense_shared_evaluator": _binding(
                "/development/dense-evaluator.py", "a"
            ),
        },
        "environment": {
            "python": "/usr/bin/python3",
            "torch": "synthetic",
            "hip": "synthetic",
            "numpy": "synthetic",
            "pillow": "synthetic",
        },
    }


def _authority_binding() -> dict[str, object]:
    return _binding("/development/authority.json", "6")


def _evaluation(*, all_pass: bool = True) -> dict[str, object]:
    gates = {
        name: {"passed": all_pass}
        for name in sorted(runner.evaluator.SCIENTIFIC_GATE_NAMES)
    }
    return {
        "schema": runner.evaluator.SCHEMA,
        "gates": gates,
        "measurement": 0.125,
    }


def _prior_compatibility_receipt(
    authority: Mapping[str, Any], stored: Mapping[str, Any]
) -> dict[str, Any]:
    recomputed = deepcopy(dict(stored))
    recomputed["measurements"]["pixels"][  # type: ignore[index]
        "minimum_reference_candidate_rgb_ssim"
    ] = 0.9998738497448542
    _, evidence = runner.compatibility.admit_task_relevance_result_v1(
        stored=stored, recomputed=recomputed
    )
    return {
        "schema": (
            "lewm_go2_dinov2_physical_readout_calibration_"
            "integrity_replacement_v1_compatibility_receipt_v1"
        ),
        "status": "PASS_PUBLISHED_BEFORE_CALIBRATION_EVAL_ACCESS",
        "citable_as_scientific_evidence": False,
        "publication_stage": (
            "inside_task_relevance_evaluator_before_outer_loader_acceptance"
        ),
        "authority_binding": _binding("/development/prior-authority", "8"),
        "preregistration_binding": _binding("/development/prior-prereg", "9"),
        "original_failure_review_binding": _binding(
            "/development/original-failure-review", "a"
        ),
        "stored_task_relevance_result_binding": authority["input_bindings"][
            "stored_task_relevance_result"
        ],
        "stored_task_relevance_review_binding": authority["input_bindings"][
            "stored_task_relevance_review"
        ],
        "task_relevance_evaluator_source_binding": _binding(
            "/development/prior-task-evaluator.py", "b"
        ),
        "environment": {"synthetic": True},
        "admission": evidence,
    }


def test_fixed_closures_and_output_inventory_match_preregistration() -> None:
    inputs = runner._fixed_input_bindings_v1()  # noqa: SLF001
    assert len(inputs) == 16
    assert set(inputs) == {
        "train_cache",
        "train_cache_receipt",
        "eval_cache",
        "eval_cache_receipt",
        "prior_calibration_result",
        "prior_calibration_terminal",
        "prior_terminal_review",
        "prior_compatibility_receipt",
        "posthoc_manifest",
        "posthoc_terminal",
        "posthoc_terminal_review",
        "posthoc_rgb_manifest",
        "posthoc_train_rows",
        "posthoc_eval_rows",
        "stored_task_relevance_result",
        "stored_task_relevance_review",
    }
    assert inputs["train_cache"]["sha256"] == (
        "164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b"
    )
    assert inputs["eval_cache"]["sha256"] == (
        "00a2e197d98effcd192392f50170648622a7210f954075002dc8b43110c636f8"
    )
    assert len(runner.SOURCE_PATHS) == 35
    assert set(runner.OUTPUT_NAMES) == {
        "reservation.json",
        "primary_compatibility_receipt.json",
        "pca_readout_checkpoint.pt",
        "evaluation.json",
        "replay_compatibility_receipt.json",
        "replay.json",
        "result.json",
        "terminal.json",
    }
    assert runner.PREREGISTRATION_SHA256 == (
        "630a1bd508629878f6eab1cd4d7839d530e6f9216789bd388f32d4853c2e3f34"
    )
    assert runner.PREREGISTRATION_BYTE_COUNT == 17_418
    assert runner.config_v1()["source_file_count"] == 35
    assert runner.config_v1()["direct_input_file_count"] == 16
    assert runner.config_v1()["compatibility_mode"] == (
        "bound_prior_evidence_replay_no_rgb_no_encoder"
    )


def test_source_closure_contains_all_new_primary_and_replay_paths() -> None:
    assert {
        "dense_shared_model",
        "dense_shared_evaluator",
        "dense_shared_runner",
        "dense_shared_replay",
        "dense_shared_model_test",
        "dense_shared_evaluator_test",
        "dense_shared_runner_test",
        "dense_shared_replay_test",
    } <= set(runner.SOURCE_PATHS)
    assert runner.SOURCE_PATHS["dense_shared_runner"] == Path(
        runner.__file__
    ).resolve()
    assert runner.SOURCE_PATHS["dense_shared_replay"] == runner.REPLAY_CLI


def test_scoped_primary_admission_publishes_before_loader_return_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _stored_document()
    calls: list[str] = []

    def live_evaluator(*_args: object, **_kwargs: object) -> Mapping[str, object]:
        calls.append("forbidden_live_evaluator")
        raise AssertionError("RGB/encoder evaluator must not run")

    def strict_loader() -> object:
        calls.append("strict_loader_enter")
        admitted = runner.task_relevance.evaluate_task_relevance_v1(
            **runner._task_relevance_call_bindings_v1(stored)  # noqa: SLF001
        )
        assert admitted is stored
        assert (tmp_path / "primary_compatibility_receipt.json").is_file()
        calls.append("strict_loader_return")
        return SimpleNamespace(bundle=True)

    monkeypatch.setattr(
        runner.task_relevance, "evaluate_task_relevance_v1", live_evaluator
    )
    monkeypatch.setattr(
        runner.prior_runner.screen_data,
        "load_bound_posthoc_bundle_v1",
        strict_loader,
    )
    monkeypatch.setattr(
        runner, "_load_stored_task_relevance_v1", lambda _authority: stored
    )
    authority = _authority(tmp_path)
    prior_receipt = _prior_compatibility_receipt(authority, stored)
    monkeypatch.setattr(
        runner,
        "_bound_document_v1",
        lambda _authority, label: (
            prior_receipt
            if label == "prior_compatibility_receipt"
            else pytest.fail(f"unexpected bound document: {label}")
        ),
    )

    with runner.scoped_primary_compatibility_admission_v1(
        authority, authority_binding=_authority_binding()
    ) as state:
        bundle = runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1()
        assert bundle.bundle is True
        assert state["evaluator_calls"] == 1
        assert state["loader_calls"] == 1
        assert state["receipt_binding"] is not None
        assert state["admission"]["differing_paths"] == [
            runner.compatibility.SSIM_DOTTED_PATH
        ]

    receipt = json.loads(
        (tmp_path / "primary_compatibility_receipt.json").read_text()
    )
    assert receipt["schema"] == runner.COMPATIBILITY_RECEIPT_SCHEMA
    assert receipt["status"] == runner.COMPATIBILITY_RECEIPT_STATUS
    assert receipt["phase"] == "primary"
    assert receipt["prior_compatibility_receipt_binding"] == authority[
        "input_bindings"
    ]["prior_compatibility_receipt"]
    assert receipt["publication_stage"] == (
        "inside_task_relevance_compatibility_replay_before_primary_"
        "strict_loader_return"
    )
    assert calls == [
        "strict_loader_enter",
        "strict_loader_return",
    ]
    assert runner.task_relevance.evaluate_task_relevance_v1 is live_evaluator
    assert (
        runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1
        is strict_loader
    )


def test_primary_loader_cannot_return_without_published_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _stored_document()
    live_evaluator = lambda *_args, **_kwargs: deepcopy(stored)
    bypass_loader = lambda: SimpleNamespace(bundle=True)
    monkeypatch.setattr(
        runner.task_relevance, "evaluate_task_relevance_v1", live_evaluator
    )
    monkeypatch.setattr(
        runner.prior_runner.screen_data,
        "load_bound_posthoc_bundle_v1",
        bypass_loader,
    )
    monkeypatch.setattr(
        runner, "_load_stored_task_relevance_v1", lambda _authority: stored
    )
    _, replayed_admission = runner.compatibility.admit_task_relevance_result_v1(
        stored=stored, recomputed=deepcopy(stored)
    )
    monkeypatch.setattr(
        runner,
        "_replay_prior_compatibility_admission_v1",
        lambda _authority, _stored: (_stored, replayed_admission),
    )

    with pytest.raises(
        runner.DenseSharedCalibrationRunnerError,
        match="not published before loader return",
    ):
        with runner.scoped_primary_compatibility_admission_v1(
            _authority(tmp_path), authority_binding=_authority_binding()
        ):
            runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1()

    assert not (tmp_path / "primary_compatibility_receipt.json").exists()
    assert runner.task_relevance.evaluate_task_relevance_v1 is live_evaluator
    assert (
        runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1
        is bypass_loader
    )


def test_prior_compatibility_replay_uses_bound_evidence_without_live_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = _stored_document()
    authority = _authority(tmp_path)
    receipt = _prior_compatibility_receipt(authority, stored)
    monkeypatch.setattr(
        runner,
        "_bound_document_v1",
        lambda _authority, label: (
            receipt
            if label == "prior_compatibility_receipt"
            else pytest.fail(f"unexpected bound document: {label}")
        ),
    )
    admitted, evidence = (  # noqa: SLF001
        runner._replay_prior_compatibility_admission_v1(
            authority, stored
        )
    )
    assert admitted is stored
    assert evidence == receipt["admission"]

    tampered = deepcopy(receipt)
    tampered["admission"][
        "recomputed_minimum_reference_candidate_rgb_ssim"
    ] = 0.9998
    monkeypatch.setattr(
        runner,
        "_bound_document_v1",
        lambda _authority, _label: tampered,
    )
    with pytest.raises(
        runner.DenseSharedCalibrationRunnerError,
        match="did not replay exactly",
    ):
        runner._replay_prior_compatibility_admission_v1(  # noqa: SLF001
            authority, stored
        )


def test_cache_wrappers_delegate_to_strict_rehashing_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = torch.empty(0, dtype=torch.float16)
    events: list[str] = []
    train_receipt = {"binding": {"synthetic": True}}

    def strict_train(bundle: object, plan: object):
        events.append(f"train:{bundle}:{plan}")
        return features, train_receipt

    monkeypatch.setattr(runner.prior_runner, "_load_train_cache_v1", strict_train)
    monkeypatch.setattr(
        runner,
        "_validate_feature_tensor_v1",
        lambda value, *, role: events.append(f"validate:{role}:{value is features}"),
    )
    assert runner._load_train_cache_v1("bundle", "plan") == (  # noqa: SLF001
        features,
        train_receipt,
    )

    authority = _authority(Path("/development/output"))
    eval_receipt = {"binding": authority["input_bindings"]["eval_cache"]}
    monkeypatch.setattr(
        runner,
        "_bound_document_v1",
        lambda _authority, label: (
            events.append(f"bound:{label}") or eval_receipt
        ),
    )

    def strict_eval(receipt: object, plan: object):
        events.append(f"eval:{receipt is eval_receipt}:{plan}")
        return features

    monkeypatch.setattr(
        runner.prior_runner, "_load_eval_feature_cache_v1", strict_eval
    )
    assert runner._load_eval_cache_v1(authority, "eval-plan") == (  # noqa: SLF001
        features,
        eval_receipt,
    )
    assert events == [
        "train:bundle:plan",
        "validate:train:True",
        "bound:eval_cache_receipt",
        "eval:True:eval-plan",
        "validate:eval:True",
    ]


def test_replay_cli_receives_only_bound_paths_hashes_and_byte_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    authority = _binding("/bound/authority.json", "a")
    checkpoint = _binding("/bound/checkpoint.pt", "b")
    evaluation = _binding("/bound/evaluation.json", "c")
    runner._launch_replay_v1(  # noqa: SLF001
        authority_binding=authority,
        checkpoint_binding=checkpoint,
        evaluation_binding=evaluation,
    )
    command = observed["command"]
    assert command[:2] == [runner.sys.executable, str(runner.REPLAY_CLI)]
    assert command[2:] == [
        "--authority",
        authority["path"],
        "--expected-authority-sha256",
        authority["sha256"],
        "--expected-authority-byte-count",
        str(authority["byte_count"]),
        "--checkpoint",
        checkpoint["path"],
        "--expected-checkpoint-sha256",
        checkpoint["sha256"],
        "--expected-checkpoint-byte-count",
        str(checkpoint["byte_count"]),
        "--evaluation",
        evaluation["path"],
        "--expected-evaluation-sha256",
        evaluation["sha256"],
        "--expected-evaluation-byte-count",
        str(evaluation["byte_count"]),
    ]
    assert observed["kwargs"]["cwd"] == runner.REPO_ROOT
    assert observed["kwargs"]["check"] is False


def test_execute_writes_checkpoint_before_eval_cache_and_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "attempt_v1"
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT_ROOT", output_root)
    authority = _authority(output_root)
    authority_binding = _authority_binding()
    bundle = SimpleNamespace(
        manifest_binding={
            "path": "/development/manifest",
            "file_sha256": "d" * 64,
            "byte_count": 1,
        }
    )
    train_groups = tuple(range(runner.ROLE_STATE_COUNT))
    eval_groups = tuple(range(runner.ROLE_STATE_COUNT, runner.ROLE_STATE_COUNT * 2))
    train_features = torch.tensor([1.0], dtype=torch.float16)
    eval_features = torch.tensor([2.0], dtype=torch.float16)
    events: list[str] = []
    primary_admission = {"differing_paths": [runner.compatibility.SSIM_DOTTED_PATH]}

    monkeypatch.setattr(runner, "_validate_prior_route_v1", lambda _authority: None)

    @contextmanager
    def admission(*_args: object, **_kwargs: object) -> Iterator[dict[str, object]]:
        runner._write_json_exclusive(  # noqa: SLF001
            output_root / "primary_compatibility_receipt.json",
            runner._compatibility_receipt_v1(  # noqa: SLF001
                phase="primary",
                authority=authority,
                authority_binding=authority_binding,
                admission=primary_admission,
            ),
        )
        yield {"admission": primary_admission}

    monkeypatch.setattr(
        runner, "scoped_primary_compatibility_admission_v1", admission
    )
    monkeypatch.setattr(
        runner.prior_runner.screen_data,
        "load_bound_posthoc_bundle_v1",
        lambda: bundle,
    )
    monkeypatch.setattr(
        runner,
        "_feature_plans_v1",
        lambda _bundle: (train_groups, eval_groups, "train-plan", "eval-plan"),
    )
    monkeypatch.setattr(
        runner,
        "_load_train_cache_v1",
        lambda _bundle, _plan: (
            train_features,
            {"schema": "train-receipt", "binding": {"synthetic": True}},
        ),
    )
    monkeypatch.setattr(runner, "_authorized_device_v1", lambda: torch.device("cpu"))

    def fit(
        *args: object, implementation_source_binding: Mapping[str, Any]
    ) -> dict[str, object]:
        events.append("fit")
        assert args == (train_groups, train_features, torch.device("cpu"))
        assert implementation_source_binding == authority["source_bindings"][
            "dense_shared_evaluator"
        ]
        return {"schema": "synthetic-checkpoint", "weight": torch.tensor([3.0])}

    monkeypatch.setattr(runner.evaluator, "fit_primary_checkpoint_v1", fit)
    original_save = runner._save_torch_exclusive  # noqa: SLF001

    def save(path: Path, payload: Mapping[str, Any]) -> None:
        events.append("checkpoint_save")
        original_save(path, payload)

    monkeypatch.setattr(runner, "_save_torch_exclusive", save)

    def load_eval(_authority: object, _plan: object):
        events.append("eval_cache_load")
        assert (output_root / "pca_readout_checkpoint.pt").is_file()
        return eval_features, {
            "schema": "eval-receipt",
            "binding": authority["input_bindings"]["eval_cache"],
        }

    monkeypatch.setattr(runner, "_load_eval_cache_v1", load_eval)
    evaluation_document = _evaluation(all_pass=True)

    def evaluate_primary(
        *args: object, implementation_source_binding: Mapping[str, Any]
    ) -> dict[str, object]:
        events.append("evaluate")
        assert args[1:] == (
            train_groups,
            eval_groups,
            train_features,
            eval_features,
            torch.device("cpu"),
        )
        assert implementation_source_binding == authority["source_bindings"][
            "dense_shared_evaluator"
        ]
        return evaluation_document

    monkeypatch.setattr(
        runner.evaluator, "evaluate_primary_checkpoint_v1", evaluate_primary
    )
    monkeypatch.setattr(
        runner, "_execution_bindings_unchanged", lambda *_args, **_kwargs: None
    )

    def launch_replay(
        *,
        authority_binding: Mapping[str, Any],
        checkpoint_binding: Mapping[str, Any],
        evaluation_binding: Mapping[str, Any],
    ) -> None:
        events.append("replay")
        assert (output_root / "evaluation.json").is_file()
        receipt = runner._compatibility_receipt_v1(  # noqa: SLF001
            phase="replay",
            authority=authority,
            authority_binding=authority_binding,
            admission=primary_admission,
        )
        runner._write_json_exclusive(  # noqa: SLF001
            output_root / "replay_compatibility_receipt.json", receipt
        )
        receipt_binding = runner.file_binding_v1(
            output_root / "replay_compatibility_receipt.json"
        )
        replay = {
            "schema": runner.REPLAY_SCHEMA,
            "status": runner.REPLAY_STATUS,
            "citable_as_scientific_evidence": False,
            "authority_binding": dict(authority_binding),
            "checkpoint_binding": dict(checkpoint_binding),
            "primary_evaluation_binding": dict(evaluation_binding),
            "compatibility_receipt_binding": receipt_binding,
            "recomputed_evaluation": evaluation_document,
            "reproduction": {
                name: True for name in runner.REPLAY_REPRODUCTION_FIELDS
            },
            "protected_material_opened": False,
            "rgb_access": {"train": 0, "eval": 0},
        }
        runner._write_json_exclusive(  # noqa: SLF001
            output_root / "replay.json", replay
        )

    monkeypatch.setattr(runner, "_launch_replay_v1", launch_replay)
    monkeypatch.setattr(
        runner.evaluator,
        "verdict_v1",
        lambda value, *, infrastructure_checks_passed, deterministic_replay_passed: {
            "gates": {
                "1_infrastructure_and_custody": {
                    "passed": infrastructure_checks_passed
                },
                **value["gates"],
                "7_deterministic_replay": {
                    "passed": deterministic_replay_passed
                },
            },
            "passed": True,
            "terminal_status": runner.PASS_STATUS,
        },
    )

    report = runner.execute_v1(
        authority, authority_binding=authority_binding
    )
    assert report["status"] == runner.PASS_STATUS
    assert events == [
        "fit",
        "checkpoint_save",
        "eval_cache_load",
        "evaluate",
        "replay",
    ]
    assert {path.name for path in output_root.iterdir()} == set(runner.OUTPUT_NAMES)
    terminal = json.loads((output_root / "terminal.json").read_text())
    assert terminal["deterministic_replay_passed"] is True
    assert terminal["failure"] is None


def test_replay_validation_rejects_nonexact_reproduction() -> None:
    evaluation = _evaluation(all_pass=True)
    authority = _authority_binding()
    checkpoint = _binding("/development/checkpoint", "7")
    primary_evaluation = _binding("/development/evaluation", "8")
    compatibility_receipt = _binding("/development/replay-receipt", "9")
    replay = {
        "schema": runner.REPLAY_SCHEMA,
        "status": runner.REPLAY_STATUS,
        "citable_as_scientific_evidence": False,
        "authority_binding": authority,
        "checkpoint_binding": checkpoint,
        "primary_evaluation_binding": primary_evaluation,
        "compatibility_receipt_binding": compatibility_receipt,
        "recomputed_evaluation": evaluation,
        "reproduction": {
            name: True for name in runner.REPLAY_REPRODUCTION_FIELDS
        },
        "protected_material_opened": False,
        "rgb_access": {"train": 0, "eval": 0},
    }
    runner._validate_replay_v1(  # noqa: SLF001
        replay,
        authority_binding=authority,
        checkpoint_binding=checkpoint,
        evaluation_binding=primary_evaluation,
        replay_compatibility_binding=compatibility_receipt,
        evaluation=evaluation,
    )
    replay["reproduction"]["state_dict_identities"] = False
    with pytest.raises(
        runner.DenseSharedCalibrationRunnerError,
        match="did not reproduce",
    ):
        runner._validate_replay_v1(  # noqa: SLF001
            replay,
            authority_binding=authority,
            checkpoint_binding=checkpoint,
            evaluation_binding=primary_evaluation,
            replay_compatibility_binding=compatibility_receipt,
            evaluation=evaluation,
        )


def test_main_writes_fail_closed_terminal_after_attempt_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "attempt_v1"
    authority = _authority(output_root)
    authority_binding = _authority_binding()
    monkeypatch.setattr(
        runner,
        "_read_authority",
        lambda *_args, **_kwargs: (authority, authority_binding),
    )

    def fail_after_reservation(*_args: object, **_kwargs: object) -> None:
        output_root.mkdir()
        runner._write_json_exclusive(  # noqa: SLF001
            output_root / "reservation.json",
            {"schema": runner.RESERVATION_SCHEMA},
        )
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runner, "execute_v1", fail_after_reservation)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.main(
            [
                "--authority",
                "/development/authority.json",
                "--expected-authority-sha256",
                "a" * 64,
                "--expected-authority-byte-count",
                "1",
            ]
        )
    terminal = json.loads((output_root / "terminal.json").read_text())
    assert terminal == {
        "schema": runner.TERMINAL_SCHEMA,
        "status": runner.FAIL_STATUS,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "authorizes_model_training": False,
        "result_binding": None,
        "deterministic_replay_passed": False,
        "failure": {
            "error_type": "RuntimeError",
            "error_message": "synthetic failure",
        },
    }


def test_source_review_requires_every_registered_check() -> None:
    preregistration = _binding("/development/preregistration", "a")
    sources = {"runner": _binding("/development/runner", "b")}
    review = {
        "schema": runner.SOURCE_REVIEW_SCHEMA,
        "status": runner.SOURCE_REVIEW_STATUS,
        "review_date": "2026-08-03",
        "reviewer": {
            "identity": "independent reviewer",
            "independence_basis": "did not author the implementation",
        },
        "protected_material_opened": False,
        "preregistration_binding": preregistration,
        "source_bindings": sources,
        "checks": {name: True for name in runner.SOURCE_REVIEW_CHECKS},
        "findings": [],
    }
    runner._validate_source_review_v1(  # noqa: SLF001
        review,
        preregistration_binding=preregistration,
        source_bindings=sources,
    )
    invalid = deepcopy(review)
    invalid["checks"].pop(next(iter(runner.SOURCE_REVIEW_CHECKS)))
    with pytest.raises(
        runner.DenseSharedCalibrationRunnerError,
        match="did not pass exactly",
    ):
        runner._validate_source_review_v1(  # noqa: SLF001
            invalid,
            preregistration_binding=preregistration,
            source_bindings=sources,
        )
