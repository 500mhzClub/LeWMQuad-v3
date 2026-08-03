from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts import run_go2_dinov2_physical_readout_calibration_v1 as runner


class _FakeDino(torch.nn.Module):
    def forward_features(self, inputs: torch.Tensor):
        batch = int(inputs.shape[0])
        values = torch.arange(1, 385, dtype=torch.float32).view(1, 1, 384)
        return {"x_norm_patchtokens": values.repeat(batch, 256, 1)}


def test_eval_extraction_opens_each_eval_artifact_once_then_replay_is_cache_only(
    tmp_path, monkeypatch
) -> None:
    plan = SimpleNamespace(role="eval", artifact_ids=("eval-0", "eval-1"))
    bundle = SimpleNamespace(manifest_binding={"path": "/synthetic/manifest"})
    authority = {"encoder_source": {"synthetic": True}}
    opened: list[str] = []

    def read_rgb(_bundle, artifact_id: str) -> bytes:
        assert _bundle is bundle
        opened.append(artifact_id)
        return artifact_id.encode("ascii")

    monkeypatch.setattr(runner, "read_bound_rgb_bytes_v1", read_rgb)
    monkeypatch.setattr(
        runner.screen_data,
        "preprocess_dinov2_png_bytes_v1",
        lambda _raw: torch.zeros(3, 2, 2, dtype=torch.float32),
    )
    monkeypatch.setattr(
        runner, "_load_dino_encoder_v1", lambda _authority, _device: _FakeDino()
    )
    output = tmp_path / "dinov2_eval.pt"
    receipt = runner.extract_eval_feature_cache_v1(
        bundle,
        plan,
        authority=authority,
        device=torch.device("cpu"),
        output_path=output,
        expected_artifact_count=2,
        batch_size=1,
    )

    assert opened == ["eval-0", "eval-1"]
    assert receipt["eval_artifact_open_count"] == 2
    assert receipt["train_artifact_open_count"] == 0
    assert receipt["decoded_pixel_verification_count"] == 2
    assert receipt["shape"] == [2, 256, 384]
    assert receipt["storage_dtype"] == "float16"

    def forbidden_rgb_open(*_args, **_kwargs):
        raise AssertionError("cache replay must not reopen RGB")

    monkeypatch.setattr(runner, "read_bound_rgb_bytes_v1", forbidden_rgb_open)
    replay = runner._load_eval_feature_cache_v1(  # noqa: SLF001
        receipt, plan, expected_artifact_count=2
    )
    assert replay.shape == (2, 256, 384)
    assert replay.dtype == torch.float16


def test_eval_extraction_rejects_train_role_before_encoder_or_rgb_access(
    tmp_path, monkeypatch
) -> None:
    plan = SimpleNamespace(role="train", artifact_ids=("train-0",))
    monkeypatch.setattr(
        runner,
        "_load_dino_encoder_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("encoder must not load for a train-role plan")
        ),
    )
    monkeypatch.setattr(
        runner,
        "read_bound_rgb_bytes_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("train RGB must not open")
        ),
    )
    with pytest.raises(runner.CalibrationRunnerError, match="eval feature plan role"):
        runner.extract_eval_feature_cache_v1(
            SimpleNamespace(manifest_binding={}),
            plan,
            authority={"encoder_source": {}},
            device=torch.device("cpu"),
            output_path=tmp_path / "forbidden.pt",
            expected_artifact_count=1,
            batch_size=1,
        )
    assert not (tmp_path / "forbidden.pt").exists()


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/sealed_test.json",
        "/tmp/sealed/member.json",
        "/tmp/sealed_future/member.json",
        "/tmp/heldout/member.json",
    ],
)
def test_protected_paths_are_rejected_before_open(path: str) -> None:
    with pytest.raises(runner.CalibrationRunnerError, match="protected material"):
        runner.file_binding_v1(runner.Path(path))


def test_frozen_train_cache_loader_is_metadata_only(monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROLE_ARTIFACT_COUNT", 2)
    plan = SimpleNamespace(role="train", artifact_ids=("train-0", "train-1"))
    index = SimpleNamespace(artifact_ids=plan.artifact_ids)
    features = torch.ones(2, 256, 384, dtype=torch.float16)
    receipt = {"encoder": "dinov2"}
    monkeypatch.setattr(runner, "_read_bound_json", lambda *_args, **_kwargs: (receipt, {}))
    monkeypatch.setattr(
        runner.predecessor_runner, "build_screen_index_v1", lambda _bundle: index
    )
    monkeypatch.setattr(
        runner.predecessor_runner,
        "_load_feature_cache",
        lambda *_args, **_kwargs: features,
    )
    monkeypatch.setattr(
        runner,
        "read_bound_rgb_bytes_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frozen train cache validation must not open RGB")
        ),
    )

    loaded, loaded_receipt = runner._load_train_cache_v1(  # noqa: SLF001
        object(), plan
    )
    assert loaded is features
    assert loaded_receipt is receipt


def test_source_closure_includes_direct_evaluator_and_predecessor_model() -> None:
    assert {
        "action_regret_evaluator",
        "counterfactual_benchmark_contract",
        "predecessor_model_module",
    } <= set(runner.SOURCE_PATHS)


def test_source_review_requires_exact_fields_and_registered_checks() -> None:
    preregistration = {"path": "/prereg", "sha256": "0" * 64, "byte_count": 1}
    sources = {
        "runner": {"path": "/runner", "sha256": "1" * 64, "byte_count": 1}
    }
    review = {
        "schema": runner.SOURCE_REVIEW_SCHEMA,
        "status": runner.SOURCE_REVIEW_STATUS,
        "review_date": "2026-08-03",
        "reviewer": "independent",
        "protected_material_opened": False,
        "preregistration_binding": preregistration,
        "source_bindings": sources,
        "checks": {name: True for name in runner.SOURCE_REVIEW_CHECKS},
        "audit_history": [],
        "findings": [],
    }
    runner._validate_source_review_v1(  # noqa: SLF001
        review,
        preregistration_binding=preregistration,
        source_bindings=sources,
    )

    placeholder = {**review, "checks": {"placeholder": True}}
    with pytest.raises(runner.CalibrationRunnerError, match="did not pass exactly"):
        runner._validate_source_review_v1(  # noqa: SLF001
            placeholder,
            preregistration_binding=preregistration,
            source_bindings=sources,
        )

    extra_field = {**review, "unreviewed": True}
    with pytest.raises(runner.CalibrationRunnerError, match="did not pass exactly"):
        runner._validate_source_review_v1(  # noqa: SLF001
            extra_field,
            preregistration_binding=preregistration,
            source_bindings=sources,
        )


def test_end_of_run_rehashes_complete_execution_closure(tmp_path, monkeypatch) -> None:
    def bound(name: str):
        path = tmp_path / name
        path.write_text(name)
        return runner.file_binding_v1(path)

    authority_binding = bound("authority.json")
    preregistration = bound("preregistration.md")
    source_review = bound("source_review.json")
    checkpoint = bound("checkpoint.pth")
    source = bound("source.py")
    fixed_input = bound("input.json")
    repo = tmp_path / "dinov2"
    repo.mkdir()
    authority = {
        "preregistration_binding": preregistration,
        "source_review_binding": source_review,
        "source_bindings": {"runner": source},
        "input_bindings": {"manifest": fixed_input},
        "encoder_source": {
            "repo_path": str(repo),
            "checkpoint_binding": checkpoint,
        },
    }

    def completed(command, **_kwargs):
        if "rev-parse" in command:
            return SimpleNamespace(stdout=runner.DINO_REPOSITORY_COMMIT + "\n")
        if "status" in command:
            return SimpleNamespace(stdout="")
        raise AssertionError(f"unexpected subprocess: {command}")

    monkeypatch.setattr(runner.subprocess, "run", completed)
    runner._execution_bindings_unchanged(  # noqa: SLF001
        authority, authority_binding=authority_binding
    )

    runner.Path(source["path"]).write_text("changed")
    with pytest.raises(runner.CalibrationRunnerError, match="source runner changed"):
        runner._execution_bindings_unchanged(  # noqa: SLF001
            authority, authority_binding=authority_binding
        )


@pytest.mark.parametrize(
    ("expected_status", "failed_scientific_gate", "replay_drift"),
    [
        (runner.PASS_STATUS, False, False),
        (runner.STOP_STATUS, True, False),
        (runner.FAIL_STATUS, False, True),
    ],
)
def test_execute_runs_two_cache_only_evaluations_and_propagates_exact_terminal(
    tmp_path,
    monkeypatch,
    expected_status: str,
    failed_scientific_gate: bool,
    replay_drift: bool,
) -> None:
    output_root = tmp_path / "attempt"
    authority = {
        "output_root": str(output_root),
        "preregistration_binding": {
            "path": "/synthetic/preregistration",
            "sha256": "0" * 64,
            "byte_count": 1,
        },
    }
    authority_binding = {
        "path": "/synthetic/authority",
        "sha256": "1" * 64,
        "byte_count": 1,
    }
    bundle = SimpleNamespace(
        access_audit={"rgb_leaf_open_count": 0},
        groups_by_role={
            "train": tuple(object() for _ in range(runner.ROLE_STATE_COUNT)),
            "eval": tuple(object() for _ in range(runner.ROLE_STATE_COUNT)),
        },
        manifest_binding={
            "path": "/synthetic/manifest",
            "sha256": "2" * 64,
            "byte_count": 1,
        },
    )
    train_plan = SimpleNamespace(
        role="train",
        artifact_ids=tuple(
            f"train-{index}" for index in range(runner.ROLE_ARTIFACT_COUNT)
        ),
    )
    eval_plan = SimpleNamespace(
        role="eval",
        artifact_ids=tuple(
            f"eval-{index}" for index in range(runner.ROLE_ARTIFACT_COUNT)
        ),
    )
    scientific_gates = {
        name: {"passed": True} for name in runner.calibration.SCIENTIFIC_GATE_NAMES
    }
    if failed_scientific_gate:
        scientific_gates["3_true_future_beats_task_action_only"] = {"passed": False}
    evaluation = {
        "schema": runner.calibration.SCHEMA,
        "gates": scientific_gates,
        "scientific_gates_2_to_6_passed": not failed_scientific_gate,
        "measurement": 1.0,
    }
    train_receipt = {"schema": "synthetic_train_cache"}
    eval_receipt = {
        "schema": runner.EVAL_CACHE_RECEIPT_SCHEMA,
        "binding": {"path": "/synthetic/cache", "sha256": "3" * 64, "byte_count": 1},
    }
    evaluation_calls = []
    eval_cache_loads = []
    extraction_calls = []
    verdict_arguments = []
    closure_checks = []

    monkeypatch.setattr(
        runner.screen_data, "load_bound_posthoc_bundle_v1", lambda: bundle
    )
    monkeypatch.setattr(
        runner.calibration,
        "build_calibration_feature_plans_v1",
        lambda *_args: SimpleNamespace(train=train_plan, eval=eval_plan),
    )
    monkeypatch.setattr(
        runner,
        "_load_train_cache_v1",
        lambda *_args: (torch.tensor([1.0]), train_receipt),
    )
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)

    def extract(*_args, **_kwargs):
        extraction_calls.append(True)
        return eval_receipt

    monkeypatch.setattr(runner, "extract_eval_feature_cache_v1", extract)

    def load_eval(*_args, **_kwargs):
        eval_cache_loads.append(True)
        return torch.tensor([2.0])

    monkeypatch.setattr(runner, "_load_eval_feature_cache_v1", load_eval)

    def evaluate(*_args):
        evaluation_calls.append(True)
        if replay_drift and len(evaluation_calls) == 2:
            return {**evaluation, "measurement": 2.0}
        return dict(evaluation)

    monkeypatch.setattr(runner, "_evaluate_v1", evaluate)
    monkeypatch.setattr(
        runner,
        "_require_binding",
        lambda value, *, label: dict(value),
    )
    monkeypatch.setattr(
        runner,
        "_execution_bindings_unchanged",
        lambda *_args, **_kwargs: closure_checks.append(True),
    )
    monkeypatch.setattr(
        runner,
        "read_bound_rgb_bytes_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("execute replay must not open RGB outside extraction")
        ),
    )
    original_verdict = runner.calibration.calibration_verdict_v1

    def verdict(selected_evaluation, **kwargs):
        verdict_arguments.append((selected_evaluation, dict(kwargs)))
        return original_verdict(selected_evaluation, **kwargs)

    monkeypatch.setattr(runner.calibration, "calibration_verdict_v1", verdict)
    report = runner.execute_v1(authority, authority_binding=authority_binding)

    assert report["status"] == expected_status
    assert len(extraction_calls) == 1
    assert len(eval_cache_loads) == 2
    assert len(evaluation_calls) == 2
    assert len(verdict_arguments) == 1
    assert verdict_arguments[0][1] == {
        "infrastructure_checks_passed": True,
        "deterministic_replay_passed": not replay_drift,
    }
    assert closure_checks == [True]
    assert runner.json.loads((output_root / "result.json").read_text())["status"] == (
        expected_status
    )
    assert runner.json.loads((output_root / "terminal.json").read_text())["status"] == (
        expected_status
    )
