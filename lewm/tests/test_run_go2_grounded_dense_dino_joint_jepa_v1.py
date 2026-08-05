from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import run_go2_grounded_dense_dino_joint_jepa_v1 as runner


def test_command_tape_is_exact_channel_major_alignment() -> None:
    block = torch.tensor(
        [
            [0.0, 10.0, 20.0],
            [1.0, 11.0, 21.0],
            [2.0, 12.0, 22.0],
            [3.0, 13.0, 23.0],
            [4.0, 14.0, 24.0],
        ]
    )
    tape = runner.command_tape_channel_major_v1(block)
    assert tape.shape == (15,)
    assert torch.equal(tape[:5], torch.arange(5, dtype=torch.float32))
    assert torch.equal(tape[5:10], torch.arange(10, 15, dtype=torch.float32))
    assert torch.equal(tape[10:], torch.arange(20, 25, dtype=torch.float32))


@pytest.mark.parametrize(
    "shape",
    [(15,), (3, 5), (5, 2), (1, 5, 3)],
)
def test_command_tape_rejects_every_noncanonical_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(runner.GroundedRunnerError, match="shape"):
        runner.command_tape_channel_major_v1(torch.zeros(shape))


def test_sampler_is_deterministic_microbatch2_accum4_and_complete_state() -> None:
    first = runner.optimizer_microbatches_v1(state_count=16, updates=4, seed=17)
    second = runner.optimizer_microbatches_v1(state_count=16, updates=4, seed=17)
    assert len(first) == 4
    assert all(len(update) == 4 for update in first)
    assert all(
        tuple(batch.state_indices.shape) == (2,) for update in first for batch in update
    )
    assert all(
        tuple(batch.candidate_action_ids.shape) == (2, 9)
        for update in first
        for batch in update
    )
    assert all(
        batch.state_indices.dtype == torch.long
        and batch.candidate_action_ids.dtype == torch.long
        for update in first
        for batch in update
    )
    assert [
        (batch.state_indices.tolist(), batch.candidate_action_ids.tolist())
        for update in first
        for batch in update
    ] == [
        (batch.state_indices.tolist(), batch.candidate_action_ids.tolist())
        for update in second
        for batch in update
    ]
    for update in first:
        selected = torch.cat([batch.state_indices for batch in update]).tolist()
        assert len(selected) == runner.EFFECTIVE_BATCH_STATES
        assert len(set(selected)) == runner.EFFECTIVE_BATCH_STATES
        for batch in update:
            assert torch.equal(
                torch.sort(batch.candidate_action_ids, dim=1).values,
                torch.arange(9).expand(2, -1),
            )
    assert any(
        not torch.equal(
            batch.candidate_action_ids,
            torch.arange(9).expand(2, -1),
        )
        for update in first
        for batch in update
    )
    # Sixteen states divide exactly into two effective batches per epoch.
    assert sorted(
        torch.cat([batch.state_indices for update in first[:2] for batch in update]).tolist()
    ) == list(range(16))
    assert runner.optimizer_schedule_identity_v1(
        first
    ) == runner.optimizer_schedule_identity_v1(second)


def test_matched_arms_receive_identical_two_group_optimizer_inventories() -> None:
    class TinyMatchedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.online_tail = torch.nn.Linear(3, 3)
            self.target_tail = deepcopy(self.online_tail).requires_grad_(False)
            self.predictor = torch.nn.Linear(3, 2)
            self.physical_head = torch.nn.Linear(2, 1)

    torch.manual_seed(9)
    physical = TinyMatchedModel()
    joint = deepcopy(physical)
    physical_optimizer = runner._optimizer_v1(physical)  # noqa: SLF001
    joint_optimizer = runner._optimizer_v1(joint)  # noqa: SLF001

    def inventory(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> list[tuple[float, list[str]]]:
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        return [
            (
                float(group["lr"]),
                [names[id(parameter)] for parameter in group["params"]],
            )
            for group in optimizer.param_groups
        ]

    expected = [
        (3.0e-5, ["online_tail.weight", "online_tail.bias"]),
        (
            3.0e-4,
            [
                "predictor.weight",
                "predictor.bias",
                "physical_head.weight",
                "physical_head.bias",
            ],
        ),
    ]
    assert inventory(physical, physical_optimizer) == expected
    assert inventory(joint, joint_optimizer) == expected
    assert all(
        torch.equal(left, right)
        for left, right in zip(physical.parameters(), joint.parameters(), strict=True)
    )


def test_access_ledger_forbids_successor_open_during_physical_arm() -> None:
    ledger = runner.AccessLedgerV1()
    ledger.load_receipts("train")
    ledger.open_rgb("train", "context", "ctx-0")
    with pytest.raises(runner.GroundedRunnerError, match="successor"):
        ledger.open_rgb("train", "successor", "future-0")
    assert ledger.rgb_opens["train_successor"] == 0

    ledger.checkpoint("physical_only_matched")
    ledger.open_rgb("train", "successor", "future-0")
    assert ledger.rgb_opens["train_successor"] == 1


def test_eval_receipts_and_context_are_blocked_until_both_checkpoints() -> None:
    ledger = runner.AccessLedgerV1()
    ledger.load_receipts("train")
    with pytest.raises(runner.GroundedRunnerError, match="before both checkpoints"):
        ledger.load_receipts("eval")
    ledger.checkpoint("physical_only_matched")
    with pytest.raises(runner.GroundedRunnerError, match="before both checkpoints"):
        ledger.load_receipts("eval")
    ledger.checkpoint("joint_jepa_grounded")
    ledger.load_receipts("eval")
    ledger.open_rgb("eval", "context", "eval-ctx")
    with pytest.raises(runner.GroundedRunnerError, match="forbidden"):
        ledger.open_rgb("eval", "successor", "eval-future")
    assert ledger.audit()["evaluation_successor_rgb_open_count"] == 0


def test_access_ledger_rejects_duplicate_rgb_open() -> None:
    ledger = runner.AccessLedgerV1()
    ledger.load_receipts("train")
    ledger.open_rgb("train", "context", "ctx")
    with pytest.raises(runner.GroundedRunnerError, match="more than once"):
        ledger.open_rgb("train", "context", "ctx")


def test_final_access_audit_requires_exact_role_and_rgb_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "STATE_COUNT", 2)
    ledger = runner.AccessLedgerV1()
    ledger.load_receipts("train")
    ledger.open_role_index("train", "/train.jsonl")
    for index in range(2):
        ledger.open_state_receipt("train", f"/train/{index}.json")
    for index in range(2 * runner.CONTEXT_COUNT):
        ledger.open_rgb("train", "context", f"context-{index}")
    ledger.checkpoint("physical_only_matched")
    for index in range(2 * runner.ACTION_COUNT):
        ledger.open_rgb("train", "successor", f"successor-{index}")
    ledger.checkpoint("joint_jepa_grounded")
    audit = runner.finalized_access_audit_v1(ledger, evaluation_opened=False)
    assert audit["state_receipt_opens"] == {"train": 2, "eval": 0}
    assert audit["evaluation_successor_rgb_open_count"] == 0


def _trace(
    *, regret: float, retrieval: float, cosine: float, persistence: float = 1.0
) -> dict[str, object]:
    return {
        "normalized_physical_rank_regret": regret,
        "branch_retrieval_accuracy": retrieval,
        "successor_cosine_error": cosine,
        "persistence_cosine_error": persistence,
        "all_finite": True,
    }


def test_futility_requires_every_registered_update400_condition() -> None:
    initial = _trace(regret=0.50, retrieval=0.11, cosine=1.0)
    passing = _trace(regret=0.47, retrieval=0.35, cosine=0.90)
    decision = runner.train_only_futility_v1(initial, passing)
    assert decision["continue_to_update_800"] is True
    assert all(decision["criteria"].values())

    for key, value in {
        "normalized_physical_rank_regret": 0.471,
        "branch_retrieval_accuracy": 0.349,
        "successor_cosine_error": 0.901,
    }.items():
        failing = dict(passing)
        failing[key] = value
        assert runner.train_only_futility_v1(initial, failing)[
            "continue_to_update_800"
        ] is False


def test_last_frame_persistence_uses_detached_ema_target_encoder() -> None:
    class TargetOnlyModel:
        def __init__(self) -> None:
            self.seen: torch.Tensor | None = None

        def encode_target(self, value: torch.Tensor) -> torch.Tensor:
            self.seen = value.clone()
            return torch.full(
                (value.shape[0], 1, runner.PATCH_TOKEN_COUNT, runner.FEATURE_DIM),
                0.25,
            )

    model = TargetOnlyModel()
    context = torch.arange(
        2 * runner.CONTEXT_COUNT * runner.FULL_TOKEN_COUNT * runner.FEATURE_DIM,
        dtype=torch.float32,
    ).reshape(2, runner.CONTEXT_COUNT, runner.FULL_TOKEN_COUNT, runner.FEATURE_DIM)
    persistence = runner.ema_last_frame_persistence_v1(model, context)
    assert model.seen is not None
    assert torch.equal(model.seen, context[:, -1:])
    assert persistence.shape == (
        2,
        runner.ACTION_COUNT,
        runner.PATCH_TOKEN_COUNT,
        runner.FEATURE_DIM,
    )
    assert persistence.requires_grad is False
    assert torch.all(persistence == 0.25)


def test_strict_determinism_is_enabled_and_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"enabled": False, "warn_only": True}
    calls: list[tuple[bool, bool]] = []

    def use_deterministic(enabled: bool, *, warn_only: bool) -> None:
        calls.append((enabled, warn_only))
        state.update(enabled=enabled, warn_only=warn_only)

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(runner.torch, "use_deterministic_algorithms", use_deterministic)
    monkeypatch.setattr(
        runner.torch,
        "are_deterministic_algorithms_enabled",
        lambda: state["enabled"],
    )
    monkeypatch.setattr(
        runner.torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: state["warn_only"],
    )
    monkeypatch.setattr(runner.torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(runner.torch.backends.cudnn, "benchmark", True)
    if hasattr(runner.torch.backends.cudnn, "allow_tf32"):
        monkeypatch.setattr(runner.torch.backends.cudnn, "allow_tf32", True)
    if hasattr(runner.torch.backends.cuda.matmul, "allow_tf32"):
        monkeypatch.setattr(runner.torch.backends.cuda.matmul, "allow_tf32", True)
    result = runner.configure_determinism_v1()
    assert calls == [(True, False)]
    assert result["torch_deterministic_algorithms"] is True
    assert result["torch_deterministic_warn_only"] is False
    assert result["cudnn_deterministic"] is True
    assert result["cudnn_benchmark"] is False
    assert result["nondeterministic_operation_policy"] == "error"


def test_live_device_must_match_the_exact_r9700_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {"device_name": "AMD Radeon AI PRO R9700"}
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        runner.torch.cuda, "get_device_name", lambda index: "AMD Radeon AI PRO R9700"
    )
    assert runner.validate_live_device_v1(environment) == environment["device_name"]

    monkeypatch.setattr(
        runner.torch.cuda, "get_device_name", lambda index: "AMD Radeon RX 7900 XTX"
    )
    with pytest.raises(runner.GroundedRunnerError, match="live device 0"):
        runner.validate_live_device_v1(environment)


def test_bootstrap_payload_uses_group_results_not_a_nonexistent_results_key() -> None:
    rows = [{"scene_id": "scene-0", "normalized_rank_regret": 0.1}]
    assert runner.report_group_results_v1(
        {"summary": {}, "group_results": rows}, label="arm"
    ) is rows
    with pytest.raises(runner.GroundedRunnerError, match="group_results"):
        runner.report_group_results_v1(
            {"summary": {}, "results": rows}, label="arm"
        )


def test_role_disjointness_checks_state_scene_and_artifact_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "STATE_COUNT", 2)
    monkeypatch.setattr(runner, "SCENE_COUNT", 2)
    train = SimpleNamespace(
        role="train",
        states=(
            SimpleNamespace(state_id="train-0", scene_id="train-scene-0"),
            SimpleNamespace(state_id="train-1", scene_id="train-scene-1"),
        ),
        artifact_ids=tuple(f"train-artifact-{index}" for index in range(24)),
    )
    evaluation = SimpleNamespace(
        role="eval",
        states=(
            SimpleNamespace(state_id="eval-0", scene_id="eval-scene-0"),
            SimpleNamespace(state_id="eval-1", scene_id="eval-scene-1"),
        ),
        artifact_ids=tuple(f"eval-artifact-{index}" for index in range(24)),
    )
    evidence = runner.assert_role_disjointness_v1(train, evaluation)
    assert all(evidence[key] is True for key in evidence if key.endswith("disjoint"))

    evaluation.states[0].state_id = "train-0"
    with pytest.raises(runner.GroundedRunnerError, match="not disjoint"):
        runner.assert_role_disjointness_v1(train, evaluation)


def test_preprocess_rgb_is_exact_shape_dtype_and_normalization() -> None:
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8)).save(
        buffer, format="PNG"
    )
    result = runner.preprocess_rgb_bytes_v1(buffer.getvalue())
    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32
    expected = (torch.ones(3) - torch.tensor(runner.IMAGENET_MEAN)) / torch.tensor(
        runner.IMAGENET_STD
    )
    assert torch.allclose(result[:, 0, 0], expected)


def test_protected_path_guards_are_component_based(tmp_path: Path) -> None:
    allowed = tmp_path / "unsealed_notes" / "artifact.json"
    runner._reject_protected(allowed, label="allowed")  # noqa: SLF001
    for relative in (
        "sealed_test.json",
        "sealed/value.json",
        "sealed_role/value.json",
        "heldout/value.json",
        "protected/value.json",
    ):
        with pytest.raises(runner.GroundedRunnerError, match="protected material"):
            runner._reject_protected(tmp_path / relative, label="candidate")  # noqa: SLF001


def test_json_and_torch_outputs_are_exclusive(tmp_path: Path) -> None:
    json_path = tmp_path / "receipt.json"
    runner._write_json_exclusive(json_path, {"value": 1})  # noqa: SLF001
    with pytest.raises(FileExistsError):
        runner._write_json_exclusive(json_path, {"value": 2})  # noqa: SLF001

    torch_path = tmp_path / "checkpoint.pt"
    runner._save_torch_exclusive(torch_path, {"value": torch.tensor([1])})  # noqa: SLF001
    with pytest.raises(FileExistsError):
        runner._save_torch_exclusive(torch_path, {"value": torch.tensor([2])})  # noqa: SLF001


def test_late_eval_bindings_are_not_rehashed_during_authority_syntax_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, bool]] = []

    def fake_require(value: object, *, label: str, rehash: bool = True) -> dict[str, object]:
        seen.append((label, rehash))
        assert isinstance(value, dict)
        return value

    monkeypatch.setattr(runner, "_require_binding", fake_require)
    # Exercise the exact late classification separately from the expensive
    # authority closure; this is the line that prevents pre-checkpoint eval I/O.
    bindings = {
        "posthoc_train_rows": {"path": "/tmp/train", "sha256": "a" * 64, "byte_count": 1},
        "posthoc_eval_rows": {"path": "/tmp/eval", "sha256": "b" * 64, "byte_count": 1},
        "eval_state_receipt_000": {"path": "/tmp/eval-state", "sha256": "c" * 64, "byte_count": 1},
    }
    for label, binding in bindings.items():
        late = label in {"posthoc_eval_rows", "eval_role_index"} or label.startswith(
            "eval_state_receipt_"
        )
        runner._require_binding(binding, label=f"input {label}", rehash=not late)  # noqa: SLF001
    assert seen == [
        ("input posthoc_train_rows", True),
        ("input posthoc_eval_rows", False),
        ("input eval_state_receipt_000", False),
    ]


def test_main_writes_consumed_failure_terminal_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "attempt"
    authority = tmp_path / "authority.json"
    authority.write_text("{}")

    monkeypatch.setattr(
        runner,
        "_load_authority_v1",
        lambda *args, **kwargs: {"output_root": str(output)},
    )

    def fail(_authority: object) -> object:
        output.mkdir()
        raise RuntimeError("sentinel failure")

    monkeypatch.setattr(runner, "execute_v1", fail)
    with pytest.raises(RuntimeError, match="sentinel failure"):
        runner.main(
            [
                "--authority",
                str(authority),
                "--expected-authority-sha256",
                "0" * 64,
                "--expected-authority-byte-count",
                "2",
            ]
        )
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE"
    assert terminal["retry_authorized"] is False
    assert terminal["result_binding"] is None


def test_authority_config_and_preregistration_binding_are_frozen() -> None:
    assert runner.runner_config_v1()["arm_order"] == [
        "physical_only_matched",
        "joint_jepa_grounded",
    ]
    assert runner.runner_config_v1()["microbatch_states"] == 2
    assert runner.runner_config_v1()["accumulation_steps"] == 4
    assert runner.runner_config_v1()["maximum_updates"] == 800
    raw = runner.PREREGISTRATION.read_bytes()
    assert len(raw) == runner.PREREGISTRATION_BYTE_COUNT
    assert hashlib.sha256(raw).hexdigest() == runner.PREREGISTRATION_SHA256
