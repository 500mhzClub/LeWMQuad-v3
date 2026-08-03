from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from scripts import run_go2_matched_branch_physical_outcome_screen_v1 as runner


def _binding(path: Path | str, character: str = "a") -> dict[str, object]:
    return {"path": str(path), "sha256": character * 64, "byte_count": 1}


def _authority(output_root: Path) -> dict[str, Any]:
    return {
        "output_root": str(output_root),
        "preregistration_binding": _binding("/development/prereg", "0"),
        "input_bindings": {
            "posthoc_manifest": _binding("/development/manifest", "1"),
            "train_cache": _binding("/development/train.pt", "2"),
            "train_cache_receipt": _binding("/development/train.json", "3"),
            "eval_cache": _binding("/development/eval.pt", "4"),
            "eval_cache_receipt": _binding("/development/eval.json", "5"),
        },
        "source_bindings": {
            "physical_outcome_evaluator": _binding(
                "/development/evaluator.py", "6"
            )
        },
    }


def _state_receipt(*, role: str, index: int) -> dict[str, Any]:
    state_id = f"{role}-state-{index}"
    return {
        "status": "PHYSICS_COMPLETE",
        "state": {
            "state_id": state_id,
            "role": role,
            "scene_id": f"{role}-scene-{index // 8}",
            "family": f"family-{index // 16}",
            "group_index": index,
            "state_index_in_scene": index % 8,
        },
        "context": {
            "context_base_pose_world_sequence": [{}, {}, {}],
            "history_executed_blocks": [[], []],
            "rgb_artifact_ids": [
                f"{state_id}:context:{context}" for context in range(3)
            ],
        },
        "branches": [
            {
                "action_id": action,
                "frame_receipt": {"artifact_id": f"{state_id}:candidate:{action}"},
            }
            for action in range(9)
        ],
    }


def test_fixed_closures_and_six_file_inventory_match_preregistration() -> None:
    inputs = runner._fixed_input_bindings_v1()  # noqa: SLF001
    assert len(inputs) == 15
    assert set(inputs) == {
        "posthoc_manifest",
        "posthoc_terminal",
        "posthoc_train_rows",
        "posthoc_eval_rows",
        "posthoc_terminal_review",
        "physics_result",
        "physics_receipt_check",
        "consumed_collection_terminal",
        "authorized_collection_plan",
        "calibration_receipt",
        "train_cache",
        "train_cache_receipt",
        "eval_cache",
        "eval_cache_receipt",
        "predecessor_dense_dino_terminal_review",
    }
    assert inputs["physics_result"]["sha256"] == (
        "25caf0a5d4c69e99559a663aa4cae96fb23ef191ccf34486804c3f2243553314"
    )
    assert inputs["train_cache"]["sha256"] == (
        "164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b"
    )
    assert inputs["eval_cache"]["sha256"] == (
        "00a2e197d98effcd192392f50170648622a7210f954075002dc8b43110c636f8"
    )
    assert runner.PREREGISTRATION_SHA256 == (
        "6b758b33948ebd621698d47ec01a892c52f473fb6bec930fcdf1cb459fd8da3f"
    )
    assert runner.PREREGISTRATION_BYTE_COUNT == 10_369
    assert set(runner.OUTPUT_NAMES) == {
        "reservation.json",
        "physical_outcome_checkpoint.pt",
        "evaluation.json",
        "replay.json",
        "result.json",
        "terminal.json",
    }
    assert runner.config_v1()["direct_input_file_count"] == 15
    assert runner.config_v1()["output_inventory"] == list(runner.OUTPUT_NAMES)
    assert runner.config_v1()["legacy_task_relevance_validation_permitted"] is False


def test_source_closure_contains_all_new_primary_and_replay_paths() -> None:
    assert {
        "physical_outcome_model",
        "physical_outcome_evaluator",
        "physical_outcome_runner",
        "physical_outcome_replay",
        "physical_outcome_model_test",
        "physical_outcome_evaluator_test",
        "physical_outcome_runner_test",
        "physical_outcome_replay_test",
    } <= set(runner.SOURCE_PATHS)
    assert runner.SOURCE_PATHS["physical_outcome_runner"] == Path(
        runner.__file__
    ).resolve()
    assert runner.SOURCE_PATHS["physical_outcome_replay"] == runner.REPLAY_CLI
    assert runner.config_v1()["source_file_count"] == len(runner.SOURCE_PATHS)


def test_protected_and_parent_traversal_paths_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(
        runner.PhysicalOutcomeScreenRunnerError, match="protected material"
    ):
        runner._safe_path(  # noqa: SLF001
            tmp_path / "sealed" / "state.json", label="synthetic", must_exist=False
        )
    with pytest.raises(
        runner.PhysicalOutcomeScreenRunnerError, match="strict relative"
    ):
        runner._resolve_receipt_binding_v1(  # noqa: SLF001
            {"path": "../escape.json", "file_sha256": "a" * 64, "byte_count": 1},
            source_root=tmp_path,
            label="synthetic receipt",
        )


def test_direct_receipt_loader_rehashes_all_256_in_frozen_plan_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "physics"
    source_root.mkdir()
    planned = []
    bindings = []
    documents: dict[str, Mapping[str, Any]] = {}
    for global_index in range(256):
        role = "train" if global_index < 128 else "eval"
        role_index = global_index if role == "train" else global_index - 128
        receipt = _state_receipt(role=role, index=role_index)
        relative = Path(role) / f"state-{role_index}.json"
        path = source_root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(receipt, sort_keys=True))
        standard = runner.file_binding_v1(path)
        bindings.append(
            {
                "path": str(relative),
                "file_sha256": standard["sha256"],
                "byte_count": standard["byte_count"],
            }
        )
        planned.append(dict(receipt["state"]))
        documents[str(path)] = receipt
    plan = {"output_root": str(source_root), "states": planned}
    physics = {
        "plan_binding": {
            "path": "/development/plan.json",
            "file_sha256": "f" * 64,
            "byte_count": 1,
        },
        "state_receipt_bindings": bindings,
    }
    authority = {
        "input_bindings": {
            "authorized_collection_plan": {
                "path": "/development/plan.json",
                "sha256": "f" * 64,
                "byte_count": 1,
            }
        }
    }
    monkeypatch.setattr(runner, "PHYSICS_ROOT", source_root)

    def bound(_authority: object, label: str) -> Mapping[str, Any]:
        return plan if label == "authorized_collection_plan" else physics

    monkeypatch.setattr(runner, "_bound_document_v1", bound)
    train, evaluation, audit = runner._load_state_receipts_v1(authority)  # noqa: SLF001
    assert len(train) == len(evaluation) == 128
    assert [item["state"]["group_index"] for item in train] == list(range(128))
    assert [item["state"]["group_index"] for item in evaluation] == list(
        range(128)
    )
    assert audit["state_receipt_open_count"] == 256
    assert audit["legacy_task_relevance_validation_called"] is False
    assert audit["rgb_leaf_open_count"] == 0
    assert len(audit["receipt_binding_identity_sha256"]) == 64


def test_artifact_order_is_context_then_nine_action_targets() -> None:
    receipts = tuple(_state_receipt(role="train", index=index) for index in range(128))
    observed = runner._expected_artifact_ids_v1(receipts, role="train")  # noqa: SLF001
    assert len(observed) == 1_536
    assert observed[:12] == tuple(
        [f"train-state-0:context:{index}" for index in range(3)]
        + [f"train-state-0:candidate:{index}" for index in range(9)]
    )
    assert len(observed) == len(set(observed))


def test_cache_loader_rehashes_safe_payload_and_exact_artifact_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = tuple(_state_receipt(role="train", index=index) for index in range(128))
    artifact_ids = runner._expected_artifact_ids_v1(  # noqa: SLF001
        receipts, role="train"
    )
    payload_path = tmp_path / "train.pt"
    torch.save(
        {
            "schema": "lewm_go2_matched_branch_successor_feature_cache_v1",
            "encoder": "dinov2",
            "artifact_ids": artifact_ids,
            "features": torch.ones((1,), dtype=torch.float16),
        },
        payload_path,
    )
    cache_binding = runner.file_binding_v1(payload_path)
    manifest = _binding(tmp_path / "manifest.json", "b")
    order_sha = hashlib.sha256(
        runner.canonical_bytes_v1(list(artifact_ids))
    ).hexdigest()
    receipt = {
        "schema": "lewm_go2_matched_branch_successor_feature_cache_receipt_v1",
        "encoder": "dinov2",
        "binding": cache_binding,
        "artifact_order_sha256": order_sha,
        "artifact_count": 1_536,
        "shape": [1_536, 256, 384],
        "storage_dtype": "float16",
        "source_bundle_manifest": {
            "path": manifest["path"],
            "file_sha256": manifest["sha256"],
            "byte_count": manifest["byte_count"],
        },
        "eval_artifact_open_count": 0,
    }
    authority = {
        "input_bindings": {
            "train_cache": cache_binding,
            "train_cache_receipt": _binding(tmp_path / "train.json", "c"),
            "posthoc_manifest": manifest,
        }
    }
    monkeypatch.setattr(
        runner,
        "_bound_document_v1",
        lambda _authority, label: receipt
        if label == "train_cache_receipt"
        else pytest.fail(f"unexpected document {label}"),
    )
    validated: list[str] = []
    monkeypatch.setattr(
        runner,
        "_validate_feature_tensor_v1",
        lambda _features, *, role: validated.append(role),
    )
    loaded, returned_receipt = runner._load_feature_cache_v1(  # noqa: SLF001
        authority, receipts, role="train"
    )
    assert loaded["artifact_ids"] == artifact_ids
    assert torch.equal(loaded["features"], torch.ones((1,), dtype=torch.float16))
    assert returned_receipt is receipt
    assert validated == ["train"]


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
    authority_binding = _binding(tmp_path / "authority.json", "a")
    train_receipts = tuple({"role": "train"} for _ in range(128))
    eval_receipts = tuple({"role": "eval"} for _ in range(128))
    events: list[str] = []
    evaluation = {
        "schema": "synthetic",
        "gates": {"synthetic": {"passed": False}},
    }
    monkeypatch.setattr(runner, "_validate_upstream_route_v1", lambda _authority: None)
    monkeypatch.setattr(
        runner,
        "_load_state_receipts_v1",
        lambda _authority: (
            train_receipts,
            eval_receipts,
            {"state_receipt_open_count": 256},
        ),
    )

    def load_cache(_authority: object, _receipts: object, *, role: str):
        if role == "eval":
            assert (output_root / "physical_outcome_checkpoint.pt").is_file()
        events.append(f"load_{role}")
        return {"artifact_ids": (), "features": torch.ones(1)}, {"role": role}

    monkeypatch.setattr(runner, "_load_feature_cache_v1", load_cache)
    monkeypatch.setattr(
        runner.evaluator,
        "build_physical_dataset_v1",
        lambda **kwargs: {"eval": kwargs["eval_receipts"] is not None},
    )
    monkeypatch.setattr(
        runner.evaluator,
        "fit_primary_checkpoint_v1",
        lambda *_args, **_kwargs: {"weight": torch.tensor([1.0])},
    )
    monkeypatch.setattr(runner.evaluator, "validate_checkpoint_v1", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner.evaluator,
        "evaluate_primary_checkpoint_v1",
        lambda *_args, **_kwargs: evaluation,
    )
    monkeypatch.setattr(
        runner.evaluator,
        "verdict_v1",
        lambda *_args, **_kwargs: {
            "gates": {"synthetic": {"passed": False}},
            "passed": False,
            "terminal_status": runner.STOP_STATUS,
        },
    )
    monkeypatch.setattr(runner, "_execution_bindings_unchanged", lambda *_a, **_k: None)

    def replay(**_kwargs: object) -> None:
        runner._write_json_exclusive(  # noqa: SLF001
            output_root / "replay.json", {"synthetic": True}
        )

    monkeypatch.setattr(runner, "_launch_replay_v1", replay)
    monkeypatch.setattr(runner, "_validate_replay_v1", lambda *_a, **_k: None)
    report = runner.execute_v1(authority, authority_binding=authority_binding)
    assert report["status"] == runner.STOP_STATUS
    assert events == ["load_train", "load_eval"]
    assert set(path.name for path in output_root.iterdir()) == set(runner.OUTPUT_NAMES)
    terminal = json.loads((output_root / "terminal.json").read_text())
    assert terminal["status"] == runner.STOP_STATUS
    assert terminal["authorizes_retry_or_resume"] is False

