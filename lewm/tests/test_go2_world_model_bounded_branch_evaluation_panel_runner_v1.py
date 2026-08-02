from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_go2_world_model_bounded_branch_evaluation_panel_v1 as runner


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "byte_count": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _probe_binding(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _analysis_document(
    *,
    result_binding: dict[str, object],
    checkpoints: dict[str, dict[str, object]],
) -> dict[str, object]:
    nested = {
        str(seed): {
            arm: checkpoints[f"{arm}/seed_{seed}"]
            for arm in runner.evaluator.MODEL_ARMS
        }
        for seed in runner.evaluator.TRAINING_SEEDS
    }
    return {
        "schema": runner.analyzer.SCHEMA,
        "status": "PASS_COMPLETE_FIXED_COMPARISON_ANALYSIS",
        "development_only": True,
        "citable_as_world_model_usefulness_evidence": False,
        "input_result": {
            "path": result_binding["path"],
            "byte_count": result_binding["byte_count"],
            "sha256": result_binding["file_sha256"],
        },
        "configuration": runner.analyzer.EXPECTED_CONFIGURATION,
        "decoder_anchor_by_seed": {},
        "contrasts": {},
        "proxy_routing": {
            "decision": "DELTA_PROXY_MEANINGFUL",
            "causal_branch_evaluation_still_required": True,
            "bulk_training_scale_authorized": False,
            "world_model_usefulness_claim_authorized": False,
        },
        "terminal_snapshot_bindings": nested,
        "uncertainty_limit": "synthetic test receipt",
    }


@pytest.fixture
def fixed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dev_root = tmp_path / "dev"
    dev_root.mkdir()
    output_root = dev_root / "go2_world_model_bounded_branch_evaluation_panel_v1"
    monkeypatch.setattr(runner, "DEV_ROOT", dev_root)
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT_ROOT", output_root)

    pilot_root = tmp_path / "pilot"
    manifest = _write(pilot_root / "manifest.json", b'{"synthetic":true}\n')
    training_root = (
        dev_root / "world_model_progression_v1" / "comparison_20260802_v1"
    )
    training_result = _write(training_root / "result.json", b'{"result":true}\n')
    terminal = _write(tmp_path / "pilot_terminal.json", b'{"terminal":true}\n')
    review = _write(tmp_path / "pilot_review.json", b'{"review":true}\n')

    checkpoint_bindings: dict[str, dict[str, object]] = {}
    for arm in runner.evaluator.MODEL_ARMS:
        for seed in runner.evaluator.TRAINING_SEEDS:
            checkpoint = (
                training_root
                / f"seed_{seed}"
                / f"{arm}_update_{runner.evaluator.EXPECTED_TERMINAL_UPDATE:06d}.pt"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{arm}/{seed}".encode("ascii"))
            checkpoint_bindings[f"{arm}/seed_{seed}"] = _probe_binding(checkpoint)

    analysis_path = training_root / "analysis.json"
    analysis_path.write_text(
        json.dumps(
            _analysis_document(
                result_binding=training_result,
                checkpoints=checkpoint_bindings,
            ),
            sort_keys=True,
        )
        + "\n"
    )
    analysis = runner.pilot.file_binding(analysis_path)
    monkeypatch.setattr(runner, "DEFAULT_PROGRESSION_ANALYSIS", analysis_path)

    manifest_receipt = {
        "path": "manifest.json",
        "file_sha256": manifest["file_sha256"],
        "byte_count": manifest["byte_count"],
    }
    groups = {
        "train": tuple(
            SimpleNamespace(scene_id=f"train_scene_{index}") for index in range(128)
        ),
        "eval": tuple(
            SimpleNamespace(scene_id=f"eval_scene_{index}") for index in range(128)
        ),
    }
    bundle = SimpleNamespace(
        manifest_binding=manifest_receipt,
        groups_by_role=groups,
    )
    terminal_gate = {
        "status": "PASS_FROZEN_BOUNDED_PILOT",
        "pilot_terminal_binding": terminal,
        "pilot_terminal_review_binding": review,
        "joined_manifest_binding": manifest_receipt,
    }
    training_separation = {
        "progression_analysis_binding": analysis,
        "training_result_binding": training_result,
        "checkpoint_panel_bindings": checkpoint_bindings,
        "scene_overlap": [],
    }
    monkeypatch.setattr(
        runner.consumer,
        "load_bound_pilot_v1",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        runner.evaluator,
        "load_and_validate_pilot_terminal_gate_v1",
        lambda *args, **kwargs: terminal_gate,
    )

    training_calls: list[dict[str, object]] = []

    def fake_training_result(*args, **kwargs):
        training_calls.append(kwargs)
        first_key = (
            f"{runner.evaluator.MODEL_ARMS[0]}/"
            f"seed_{runner.evaluator.TRAINING_SEEDS[0]}"
        )
        assert Path(kwargs["selected_checkpoint"]) == Path(
            checkpoint_bindings[first_key]["path"]
        )
        return {"status": "synthetic"}, training_separation

    monkeypatch.setattr(
        runner.evaluator,
        "load_and_validate_progression_analysis_v1",
        fake_training_result,
        raising=False,
    )

    kwargs = {
        "pilot_root": pilot_root,
        "manifest_byte_count": manifest["byte_count"],
        "manifest_sha256": manifest["file_sha256"],
        "progression_analysis": analysis_path,
        "progression_analysis_sha256": analysis["file_sha256"],
        "progression_analysis_byte_count": analysis["byte_count"],
        "pilot_terminal": Path(terminal["path"]),
        "pilot_terminal_sha256": terminal["file_sha256"],
        "pilot_terminal_byte_count": terminal["byte_count"],
        "pilot_terminal_review": Path(review["path"]),
        "pilot_terminal_review_sha256": review["file_sha256"],
        "pilot_terminal_review_byte_count": review["byte_count"],
    }
    return SimpleNamespace(
        kwargs=kwargs,
        output_root=output_root,
        checkpoint_bindings=checkpoint_bindings,
        manifest_receipt=manifest_receipt,
        terminal_gate=terminal_gate,
        training_separation=training_separation,
        training_calls=training_calls,
        analysis_path=analysis_path,
    )


def _member_report(fixed_inputs, *, arm: str, seed: int) -> dict[str, object]:
    key = f"{arm}/seed_{seed}"
    return {
        "schema": runner.evaluator.REPORT_SCHEMA,
        "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "scientific_verdict_emitted": False,
        "pilot_manifest_binding": fixed_inputs.manifest_receipt,
        "pilot_terminal_gate": fixed_inputs.terminal_gate,
        "checkpoint_binding": fixed_inputs.checkpoint_bindings[key],
        "checkpoint_panel_identity": {
            "arm": arm,
            "seed": seed,
            "update": runner.evaluator.EXPECTED_TERMINAL_UPDATE,
        },
        "training_scene_separation": fixed_inputs.training_separation,
        "source_bindings": [],
        "evaluation_contract": runner.evaluator.evaluation_contract_v1(),
        "checkpoint_gate_status": "CHECKPOINT_MEASUREMENT_FAILS_PREREGISTERED_GATES",
    }


def _complete_aggregate() -> dict[str, object]:
    return {
        "schema": runner.evaluator.PANEL_REPORT_SCHEMA,
        "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "all_fixed_panel_members_reported": True,
        "global_verdict": "USEFUL_SCENE_DISJOINT_PLANNING_EVIDENCE_NOT_ESTABLISHED",
    }


def test_exact_panel_is_reserved_before_models_and_aggregated_once(
    fixed_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_evaluate(**kwargs):
        assert (fixed_inputs.output_root / runner.RESERVATION_NAME).is_file()
        assert not (fixed_inputs.output_root / runner.PANEL_RESULT_NAME).exists()
        calls.append((kwargs["expected_arm"], kwargs["expected_training_seed"]))
        return _member_report(
            fixed_inputs,
            arm=kwargs["expected_arm"],
            seed=kwargs["expected_training_seed"],
        )

    aggregate_calls: list[list[dict[str, object]]] = []

    def fake_aggregate(reports):
        aggregate_calls.append(reports)
        assert len(reports) == 12
        assert all("global_verdict" not in report for report in reports)
        return _complete_aggregate()

    monkeypatch.setattr(runner.evaluator, "evaluate_bound_model_v1", fake_evaluate)
    monkeypatch.setattr(runner.evaluator, "aggregate_model_panel_v1", fake_aggregate)

    result = runner.run_panel_v1(**fixed_inputs.kwargs)

    assert calls == [
        (arm, seed)
        for arm in runner.evaluator.MODEL_ARMS
        for seed in runner.evaluator.TRAINING_SEEDS
    ]
    assert len(aggregate_calls) == 1
    assert len(result["member_report_bindings"]) == 12
    assert len(fixed_inputs.training_calls) == 1
    reservation = json.loads(
        (fixed_inputs.output_root / runner.RESERVATION_NAME).read_text()
    )
    assert reservation["input_bindings"]["progression_analysis"]["path"] == str(
        fixed_inputs.analysis_path.resolve()
    )
    assert reservation["retry_authorized"] is False
    assert reservation["resume_authorized"] is False
    terminal = json.loads((fixed_inputs.output_root / runner.TERMINAL_NAME).read_text())
    assert terminal["status"] == "COMPLETE_PENDING_INDEPENDENT_REVIEW"
    assert terminal["all_fixed_panel_members_reported"] is True
    assert terminal["scientific_verdict_emitted_by_terminal"] is False
    assert terminal["terminal_rehash"]["status"] == "PASS"
    reservation_check = terminal["terminal_rehash"]["checked_bindings"][
        "reservation"
    ]
    assert reservation_check["binding"] == terminal["reservation_binding"]
    assert reservation_check["schema"] == runner.PANEL_RESERVATION_SCHEMA
    assert reservation_check["status"] == "RESERVED_PANEL_ATTEMPT_CONSUMED"
    assert reservation_check["supervisor_nonce"] == reservation["supervisor_nonce"]
    assert reservation_check["supervisor_pid"] == reservation["supervisor_pid"]


def test_failed_member_is_terminal_and_fixed_root_cannot_retry(
    fixed_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    aggregate_calls = 0

    def fake_evaluate(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic member failure")
        return _member_report(
            fixed_inputs,
            arm=kwargs["expected_arm"],
            seed=kwargs["expected_training_seed"],
        )

    def fake_aggregate(reports):
        nonlocal aggregate_calls
        aggregate_calls += 1
        return _complete_aggregate()

    monkeypatch.setattr(runner.evaluator, "evaluate_bound_model_v1", fake_evaluate)
    monkeypatch.setattr(runner.evaluator, "aggregate_model_panel_v1", fake_aggregate)

    with pytest.raises(runner.EvaluationPanelRunnerError, match="retry/resume"):
        runner.run_panel_v1(**fixed_inputs.kwargs)
    terminal = json.loads((fixed_inputs.output_root / runner.TERMINAL_NAME).read_text())
    assert terminal["status"] == "FAILED_TERMINAL_NO_RETRY"
    assert terminal["failure"]["completed_members"] == 1
    assert terminal["failure"]["aggregate_written"] is False
    assert aggregate_calls == 0

    with pytest.raises(runner.EvaluationPanelRunnerError, match="already consumed"):
        runner.run_panel_v1(**fixed_inputs.kwargs)
    assert calls == 2
    assert aggregate_calls == 0


def test_member_cannot_emit_global_usefulness(
    fixed_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    aggregate_calls = 0

    def fake_evaluate(**kwargs):
        report = _member_report(
            fixed_inputs,
            arm=kwargs["expected_arm"],
            seed=kwargs["expected_training_seed"],
        )
        report["global_verdict"] = "USEFUL"
        return report

    def fake_aggregate(reports):
        nonlocal aggregate_calls
        aggregate_calls += 1
        return _complete_aggregate()

    monkeypatch.setattr(runner.evaluator, "evaluate_bound_model_v1", fake_evaluate)
    monkeypatch.setattr(runner.evaluator, "aggregate_model_panel_v1", fake_aggregate)

    with pytest.raises(runner.EvaluationPanelRunnerError, match="retry/resume"):
        runner.run_panel_v1(**fixed_inputs.kwargs)
    terminal = json.loads((fixed_inputs.output_root / runner.TERMINAL_NAME).read_text())
    assert terminal["failure"]["completed_members"] == 0
    assert aggregate_calls == 0
    assert not (fixed_inputs.output_root / runner.MEMBER_DIRECTORY).exists()


def test_analysis_must_bind_exact_existing_terminal_snapshots(
    fixed_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = json.loads(fixed_inputs.analysis_path.read_text())
    seed = str(runner.evaluator.TRAINING_SEEDS[0])
    arm = runner.evaluator.MODEL_ARMS[0]
    document["terminal_snapshot_bindings"][seed][arm]["sha256"] = "0" * 64
    fixed_inputs.analysis_path.write_text(json.dumps(document, sort_keys=True) + "\n")
    changed = runner.pilot.file_binding(fixed_inputs.analysis_path)
    fixed_inputs.kwargs["progression_analysis_sha256"] = changed["file_sha256"]
    fixed_inputs.kwargs["progression_analysis_byte_count"] = changed["byte_count"]
    fixed_inputs.training_separation["progression_analysis_binding"] = changed
    model_calls = 0

    def fake_evaluate(**kwargs):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("model evaluation must not start")

    monkeypatch.setattr(runner.evaluator, "evaluate_bound_model_v1", fake_evaluate)
    with pytest.raises(
        runner.EvaluationPanelRunnerError,
        match="checkpoint bytes changed",
    ):
        runner.run_panel_v1(**fixed_inputs.kwargs)
    assert model_calls == 0
    assert not fixed_inputs.output_root.exists()


def test_terminal_rehash_detects_reservation_tampering(
    fixed_inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered = False

    def fake_evaluate(**kwargs):
        nonlocal tampered
        if not tampered:
            reservation_path = fixed_inputs.output_root / runner.RESERVATION_NAME
            document = json.loads(reservation_path.read_text())
            document["retry_authorized"] = True
            reservation_path.write_text(json.dumps(document, sort_keys=True) + "\n")
            tampered = True
        return _member_report(
            fixed_inputs,
            arm=kwargs["expected_arm"],
            seed=kwargs["expected_training_seed"],
        )

    monkeypatch.setattr(runner.evaluator, "evaluate_bound_model_v1", fake_evaluate)
    monkeypatch.setattr(
        runner.evaluator,
        "aggregate_model_panel_v1",
        lambda reports: _complete_aggregate(),
    )

    with pytest.raises(runner.EvaluationPanelRunnerError, match="retry/resume"):
        runner.run_panel_v1(**fixed_inputs.kwargs)
    terminal = json.loads((fixed_inputs.output_root / runner.TERMINAL_NAME).read_text())
    assert terminal["status"] == "FAILED_TERMINAL_NO_RETRY"
    assert terminal["terminal_rehash"]["status"] == "FAIL"
    assert terminal["terminal_rehash"]["checked_bindings"]["reservation"] is None
    assert any(
        failure.startswith("reservation:")
        for failure in terminal["terminal_rehash"]["failures"]
    )


@pytest.mark.parametrize(
    "fabrication",
    (
        "schema",
        "nonce",
        "pid",
        "attempt_status",
        "input_bindings",
        "retry",
        "resume",
        "output_root",
        "extra_key",
    ),
)
def test_matching_binding_cannot_legitimize_fabricated_reservation(
    fixed_inputs, fabrication: str
) -> None:
    preflight = {
        "input_bindings": {
            "pilot_manifest": runner.pilot.file_binding(
                Path(fixed_inputs.kwargs["pilot_root"]) / "manifest.json"
            )
        },
        "terminal_gate": fixed_inputs.terminal_gate,
        "training_separation": fixed_inputs.training_separation,
        "checkpoint_bindings": fixed_inputs.checkpoint_bindings,
        "source_bindings": [],
    }
    output, _original_binding, expected = runner._reserve_panel_v1(
        output_root=fixed_inputs.output_root,
        preflight=preflight,
    )
    fabricated = json.loads(json.dumps(expected))
    if fabrication == "schema":
        fabricated["schema"] = "fabricated"
    elif fabrication == "nonce":
        fabricated["supervisor_nonce"] = "g" * 64
    elif fabrication == "pid":
        fabricated["supervisor_pid"] += 1
    elif fabrication == "attempt_status":
        fabricated["status"] = "UNCONSUMED"
    elif fabrication == "input_bindings":
        fabricated["input_bindings"] = {}
    elif fabrication == "retry":
        fabricated["retry_authorized"] = True
    elif fabrication == "resume":
        fabricated["resume_authorized"] = True
    elif fabrication == "output_root":
        fabricated["output_root"] = str(output / "different_attempt")
    elif fabrication == "extra_key":
        fabricated["fabricated"] = True
    reservation_path = output / runner.RESERVATION_NAME
    reservation_path.write_text(json.dumps(fabricated, sort_keys=True) + "\n")
    fabricated_binding = runner.pilot.file_binding(reservation_path)

    with pytest.raises(
        runner.EvaluationPanelRunnerError,
        match="exact consumed no-retry attempt",
    ):
        runner._validate_terminal_reservation_v1(
            output_root=output,
            reservation_binding=fabricated_binding,
            expected_reservation=fabricated,
            preflight=preflight,
        )
