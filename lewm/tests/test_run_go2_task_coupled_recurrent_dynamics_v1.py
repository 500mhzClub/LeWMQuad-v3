from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import run_go2_task_coupled_recurrent_dynamics_v1 as runner


def _write(path: Path, value: bytes) -> Path:
    path.write_bytes(value)
    return path


def _write_json(path: Path, value: object) -> Path:
    return _write(
        path,
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n",
    )


def _permissions() -> dict[str, bool]:
    return {
        "train_receipt_access": True,
        "train_context_rgb_access": True,
        "eval_receipt_access_after_checkpoint": True,
        "eval_context_rgb_access_after_checkpoint": True,
        "successor_rgb_access": False,
        "data_generation": False,
        "sealed_or_protected_access": False,
        "retry_resume_overwrite": False,
    }


def test_source_inventory_and_frozen_runner_constants_are_complete() -> None:
    assert runner.AUTHORITY_SCHEMA == (
        "lewm_go2_task_coupled_recurrent_dynamics_v1_execution_authority_v1"
    )
    assert runner.AUTHORITY_STATUS == (
        "AUTHORIZED_ONE_TASK_COUPLED_RECURRENT_DYNAMICS_V1"
    )
    assert runner.RESULT_SCHEMA == (
        "lewm_go2_task_coupled_recurrent_dynamics_v1_result_v1"
    )
    assert runner.TERMINAL_SCHEMA == (
        "lewm_go2_task_coupled_recurrent_dynamics_v1_terminal_v1"
    )
    assert runner.DEFAULT_OUTPUT_ROOT.name == "attempt_v1"
    assert {
        "recurrent_model",
        "recurrent_benchmark",
        "recurrent_runner",
        "recurrent_model_test",
        "recurrent_benchmark_test",
        "recurrent_runner_test",
    } <= set(runner.SOURCE_PATHS)
    assert runner.SOURCE_PATHS["recurrent_runner"] == Path(runner.__file__).resolve()
    assert runner.SOURCE_PATHS["recurrent_runner_test"] == Path(__file__).resolve()
    config = runner.benchmark.config_v1()
    assert config["schema"] == "lewm_go2_task_coupled_recurrent_dynamics_v1"
    assert config["states_per_role"] == 128
    assert config["actions"] == 9
    assert config["context_steps"] == 3
    assert config["arms"] == [
        runner.benchmark.NO_VISION_ARM,
        runner.benchmark.VISUAL_ARM,
    ]
    assert config["successor_observation_access"] is False


def test_context_only_ledger_enforces_checkpoint_order_and_rejects_successors() -> None:
    ledger = runner.ContextOnlyLedgerV1()

    with pytest.raises(runner.RecurrentRunnerError, match="before checkpoint"):
        ledger.load_receipts("eval")
    with pytest.raises(runner.RecurrentRunnerError, match="outside its stage"):
        ledger.open_role_index("train", "/tmp/train.jsonl")

    ledger.load_receipts("train")
    ledger.open_role_index("train", "/tmp/train.jsonl")
    ledger.open_state_receipt("train", "/tmp/train-state-0.json")
    ledger.open_rgb("train", "context", "train-context-0")
    with pytest.raises(runner.RecurrentRunnerError, match="structurally forbidden"):
        ledger.open_rgb("train", "successor", "train-future-0")
    with pytest.raises(runner.RecurrentRunnerError, match="more than once"):
        ledger.open_rgb("train", "context", "train-context-0")
    with pytest.raises(runner.RecurrentRunnerError, match="open first"):
        ledger.load_receipts("train")

    ledger.checkpoint()
    with pytest.raises(runner.RecurrentRunnerError, match="outside train stage"):
        ledger.open_rgb("train", "context", "late-train-context")
    ledger.load_receipts("eval")
    ledger.open_role_index("eval", "/tmp/eval.jsonl")
    ledger.open_state_receipt("eval", "/tmp/eval-state-0.json")
    ledger.open_rgb("eval", "context", "eval-context-0")
    with pytest.raises(runner.RecurrentRunnerError, match="structurally forbidden"):
        ledger.open_rgb("eval", "successor", "eval-future-0")

    assert ledger.rgb_opens["train_successor"] == 0
    assert ledger.rgb_opens["eval_successor"] == 0
    with pytest.raises(runner.RecurrentRunnerError, match="accounting changed"):
        ledger.finalized()


def test_context_only_ledger_finalizes_only_the_full_registered_context_inventory() -> None:
    ledger = runner.ContextOnlyLedgerV1()
    ledger.load_receipts("train")
    ledger.open_role_index("train", "/tmp/train.jsonl")
    for index in range(128):
        ledger.open_state_receipt("train", f"/tmp/train-state-{index}.json")
    for index in range(384):
        ledger.open_rgb("train", "context", f"train-context-{index}")

    ledger.checkpoint()
    ledger.load_receipts("eval")
    ledger.open_role_index("eval", "/tmp/eval.jsonl")
    for index in range(128):
        ledger.open_state_receipt("eval", f"/tmp/eval-state-{index}.json")
    for index in range(384):
        ledger.open_rgb("eval", "context", f"eval-context-{index}")

    audit = ledger.finalized()

    assert audit == {
        "stage": "eval",
        "checkpoint_durable": True,
        "receipt_loads": {"train": 1, "eval": 1},
        "role_index_opens": {"train": 1, "eval": 1},
        "state_receipt_opens": {"train": 128, "eval": 128},
        "rgb_opens": {
            "train_context": 384,
            "train_successor": 0,
            "eval_context": 384,
            "eval_successor": 0,
        },
        "unique_context_artifacts": 768,
        "successor_rgb_open_count": 0,
    }


def test_result_identity_is_canonical_and_excludes_only_its_identity_field() -> None:
    first = {
        "schema": runner.RESULT_SCHEMA,
        "status": "sentinel",
        "nested": {"b": [2, 1], "a": True},
    }
    reordered = {
        "nested": {"a": True, "b": [2, 1]},
        "status": "sentinel",
        "schema": runner.RESULT_SCHEMA,
        "result_identity_sha256": "not part of the identity",
    }

    identity = runner._result_identity_v1(first)  # noqa: SLF001

    assert len(identity) == 64
    assert identity == runner._result_identity_v1(reordered)  # noqa: SLF001
    changed = dict(first)
    changed["status"] = "changed"
    assert runner._result_identity_v1(changed) != identity  # noqa: SLF001


def _temporary_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    permission_change: tuple[str, bool] | None = None,
) -> tuple[Path, dict[str, object], dict[str, Any]]:
    preregistration = _write(tmp_path / "preregistration.md", b"frozen protocol\n")
    source = _write(tmp_path / "runner_source.py", b"# source witness\n")
    dino_checkpoint = _write(tmp_path / "dino_checkpoint.pt", b"DINO witness\n")
    dino_repository = tmp_path / "dino_repository"
    dino_repository.mkdir()
    train_input = _write(tmp_path / "train_rows.jsonl", b"train declaration\n")
    eval_input = _write(tmp_path / "eval_rows.jsonl", b"eval declaration\n")

    source_bindings = {"runner": runner.file_binding_v1(source)}
    preregistration_binding = runner.file_binding_v1(preregistration)
    input_bindings = {
        "posthoc_train_rows": runner.file_binding_v1(train_input),
        "posthoc_eval_rows": runner.file_binding_v1(eval_input),
    }
    dino_binding = runner.file_binding_v1(dino_checkpoint)
    review = {
        "schema": runner.SOURCE_REVIEW_SCHEMA,
        "status": runner.SOURCE_REVIEW_STATUS,
        "protected_material_opened": False,
        "preregistration_binding": preregistration_binding,
        "source_bindings": source_bindings,
        "findings": [],
    }
    review_path = _write_json(tmp_path / "source_review.json", review)

    output_root = tmp_path / "attempt_v1"
    monkeypatch.setattr(runner, "PREREGISTRATION", preregistration)
    monkeypatch.setattr(runner, "SOURCE_REVIEW", review_path)
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(runner, "DINO_REPOSITORY", dino_repository)
    monkeypatch.setattr(runner, "DINO_CHECKPOINT", dino_checkpoint)
    monkeypatch.setattr(runner, "SOURCE_PATHS", {"runner": source})
    monkeypatch.setattr(
        runner.upstream, "fixed_input_bindings_v1", lambda: input_bindings
    )
    monkeypatch.setattr(
        runner.upstream, "DINO_CHECKPOINT_SHA256", dino_binding["sha256"]
    )
    monkeypatch.setattr(
        runner.upstream,
        "DINO_CHECKPOINT_BYTE_COUNT",
        dino_binding["byte_count"],
    )

    permissions = _permissions()
    if permission_change is not None:
        permissions[permission_change[0]] = permission_change[1]
    authority = {
        "schema": runner.AUTHORITY_SCHEMA,
        "status": runner.AUTHORITY_STATUS,
        "output_root": str(output_root.resolve()),
        "config": runner.benchmark.config_v1(),
        "preregistration_binding": preregistration_binding,
        "source_review_binding": runner.file_binding_v1(review_path),
        "source_bindings": source_bindings,
        "input_bindings": input_bindings,
        "dino": {
            "repository_path": str(dino_repository.resolve()),
            "repository_commit": runner.upstream.DINO_REPOSITORY_COMMIT,
            "checkpoint_binding": dino_binding,
        },
        "permissions": permissions,
    }
    authority_path = _write_json(tmp_path / "authority.json", authority)
    authority_binding = runner.file_binding_v1(authority_path)
    # Authority validation must not open the evaluation role.  Its exact
    # declaration remains bound, but the file becomes available only later.
    eval_input.unlink()
    return authority_path, authority_binding, authority


def test_authority_accepts_exact_permissions_without_opening_late_eval_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_path, binding, expected = _temporary_authority(tmp_path, monkeypatch)

    observed, observed_binding = runner._validate_authority_v1(  # noqa: SLF001
        authority_path,
        expected_sha256=str(binding["sha256"]),
        expected_byte_count=int(binding["byte_count"]),
    )

    assert observed == expected
    assert observed_binding == binding
    assert observed["permissions"] == _permissions()


@pytest.mark.parametrize(
    ("permission", "changed"),
    (
        ("successor_rgb_access", True),
        ("data_generation", True),
        ("sealed_or_protected_access", True),
        ("retry_resume_overwrite", True),
        ("eval_context_rgb_access_after_checkpoint", False),
    ),
)
def test_authority_rejects_every_permission_expansion_or_contraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    permission: str,
    changed: bool,
) -> None:
    authority_path, binding, _expected = _temporary_authority(
        tmp_path,
        monkeypatch,
        permission_change=(permission, changed),
    )

    with pytest.raises(runner.RecurrentRunnerError, match="permissions changed"):
        runner._validate_authority_v1(  # noqa: SLF001
            authority_path,
            expected_sha256=str(binding["sha256"]),
            expected_byte_count=int(binding["byte_count"]),
        )


def test_binding_syntax_check_can_defer_io_but_still_rejects_malformed_values(
    tmp_path: Path,
) -> None:
    absent = tmp_path / "late_eval_rows.jsonl"
    declared = {"path": str(absent), "sha256": "a" * 64, "byte_count": 1}

    assert runner._require_binding(  # noqa: SLF001
        declared, label="late evaluation input", rehash=False
    ) == declared
    with pytest.raises(runner.RecurrentRunnerError, match="malformed"):
        runner._require_binding(  # noqa: SLF001
            {**declared, "byte_count": 0},
            label="late evaluation input",
            rehash=False,
        )
    with pytest.raises(RuntimeError):
        runner._require_binding(  # noqa: SLF001
            declared, label="late evaluation input", rehash=True
        )
