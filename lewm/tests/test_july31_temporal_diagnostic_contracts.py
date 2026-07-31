from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import dev_eval_temporal_retention_and_rollout as retention
from scripts import dev_temporal_v1_rank_asymptote_probe as rank_probe
from scripts import dev_train_temporal_jepa_scaled as scaled_trainer


def test_scaled_wrong_action_control_uses_canonical_hold_six() -> None:
    actions = torch.tensor(
        [[0, 1, 5], [1, 2, 6], [2, 3, 8]],
        dtype=torch.long,
    )

    wrong, eligible = scaled_trainer.build_wrong_action_control(actions)

    assert scaled_trainer.HOLD_ACTION == 6
    assert scaled_trainer.HOLD_ACTION == scaled_trainer.metrics.HOLD_ACTION_ID
    assert wrong[:, 2].tolist() == [6, 7, 0]
    assert eligible.tolist() == [True, False, True]
    assert torch.equal(wrong[:, :2], actions[:, :2])


def test_rank_reference_is_restricted_and_one_pass_only() -> None:
    generator = torch.Generator().manual_seed(20260731)
    source = torch.randn(8, 3, 4, generator=generator)
    actions = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]), num_classes=2
    ).float()
    action_effect = actions[:, 1].view(8, 1, 1)
    target = 0.75 * source + action_effect

    reference = rank_probe.restricted_linear_reference_rank(
        source, target, action_onehot=actions
    )

    assert set(reference) == {
        "effective_rank",
        "cross_sample_variance",
        "second_half_panel_mse",
        "fit_rows",
        "evaluation_rows",
        "split_rule",
        "stratified",
    }
    assert reference["fit_rows"] == 4
    assert reference["evaluation_rows"] == 4
    assert reference["split_rule"] == "ordered_first_half_fit_second_half_evaluation"
    assert reference["stratified"] is False
    assert reference["second_half_panel_mse"] < 1e-12
    assert rank_probe.validate_requested_updates(400) == 400
    for invalid in (True, 0, 401):
        with pytest.raises(ValueError, match="cycling is forbidden"):
            rank_probe.validate_requested_updates(invalid)

    source_text = Path(rank_probe.__file__).read_text()
    assert "linear_oracle" not in source_text
    assert "restricted_linear_reference" in source_text


def test_scaled_snapshot_is_audit_only_not_a_resume_payload() -> None:
    model = torch.nn.Linear(2, 2)
    args = SimpleNamespace(tag="synthetic", updates=1)

    payload = scaled_trainer._snapshot_payload(
        model=model,
        update=1,
        args=args,
        pack_bindings={"train": {}, "val": {}},
        predecessor_binding={"sha256": "0" * 64},
        source_bindings=[],
    )

    assert payload["schema"] == scaled_trainer.SNAPSHOT_SCHEMA
    assert payload["authorizes_retry_or_resume"] is False
    assert payload["update"] == 1
    assert "optimizer_state_dict" not in payload
    assert "shuffle_cursor" not in payload
    assert "torch_cpu_rng_state" not in payload

    trainer_source = Path(scaled_trainer.__file__).read_text()
    assert "latest.pt" not in trainer_source
    assert "final_trace.json" in trainer_source


def test_scaled_batches_carry_the_permutation_tail() -> None:
    order = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    fresh = torch.tensor([0, 3, 2, 1, 4], dtype=torch.long)

    batch, next_order, cursor = scaled_trainer.carry_permutation_tail(
        order, 4, 3, fresh_order=fresh
    )

    assert batch.tolist() == [4, 0, 3]
    assert torch.equal(next_order, fresh)
    assert cursor == 2


def test_scaled_stop_does_not_move_declared_cosine_horizon() -> None:
    args = SimpleNamespace(
        updates=700,
        schedule_updates=3000,
        batch=256,
        microbatch=32,
        lr_scale=4.0,
        warmup=150,
        eval_every=100,
        tag="bounded_u700",
        pack_root=str(scaled_trainer.DEV_OUTPUT_ROOT / "pack"),
        output_root=str(scaled_trainer.DEV_OUTPUT_ROOT / "train"),
    )

    scaled_trainer._validate_main_args(args)

    bounded = scaled_trainer.learning_rate_fraction(
        700, warmup_updates=150, schedule_updates=3000
    )
    moved_endpoint = scaled_trainer.learning_rate_fraction(
        700, warmup_updates=150, schedule_updates=700
    )
    assert bounded == pytest.approx(0.9108889, abs=1e-6)
    assert moved_endpoint == pytest.approx(0.0)

    args.schedule_updates = 699
    with pytest.raises(ValueError, match="cannot be smaller"):
        scaled_trainer._validate_main_args(args)


def test_retention_selection_is_explicit_and_hash_bound() -> None:
    parser = retention.build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--checkpoint", "snapshot.pt", "--migrated-predecessor-baseline"]
        )

    missing_hash = parser.parse_args(
        ["--checkpoint", "snapshot.pt", "--expected-update", "200"]
    )
    with pytest.raises(ValueError, match="expected-checkpoint-sha256"):
        retention.validate_selection(missing_hash)

    selected = parser.parse_args(
        [
            "--checkpoint",
            "snapshot.pt",
            "--expected-update",
            "200",
            "--expected-checkpoint-sha256",
            "a" * 64,
        ]
    )
    path, update = retention.validate_selection(selected)
    assert path == Path("snapshot.pt")
    assert update == 200

    baseline = parser.parse_args(["--migrated-predecessor-baseline"])
    assert retention.validate_selection(baseline) == (None, 0)


def test_retention_checkpoint_cannot_escape_development_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="development checkpoint must remain"):
        retention.require_development_checkpoint(tmp_path / "snapshot.pt")

    inside = retention.DEV_OUTPUT_ROOT / "temporal" / "snapshot.pt"
    assert retention.require_development_checkpoint(inside) == inside


def test_checkpoint_loads_are_safe_and_predecessor_is_pinned() -> None:
    for module in (scaled_trainer, rank_probe, retention):
        source = Path(module.__file__).read_text()
        assert "weights_only=False" not in source
        assert "weights_only=True" in source
        assert module.PREDECESSOR_BYTE_COUNT == 52_282_877
        assert module.PREDECESSOR_SHA256 == (
            "f5aac23cf275d73b92ce5609a583dea89"
            "f6686a624d4889d9762740535aab873"
        )


def test_structural_report_does_not_claim_a_rollout_measurement() -> None:
    model = SimpleNamespace(config=SimpleNamespace(spatial_token_count=256))

    report = retention.structural_composability(model)

    assert report["direct_output_to_input_plug_compatible"] is False
    assert report["overall_composability"] == "UNDETERMINED"
    assert report["adapter_or_completion_path_evaluated"] is False
    assert report["rollout_was_executed"] is False
    assert report["diagnostic_kind"] == "source_and_shape_contract_only"
    assert report["prediction_output_tokens"] == 64


def test_retention_receipt_binds_the_spatial_panel_implementation() -> None:
    paths = {path.resolve() for path in retention.retention_source_paths()}

    assert Path(retention.evaluation.spatial_evaluation.__file__).resolve() in paths
    assert (
        Path(retention.evaluation.spatial_evaluation.metrics.__file__).resolve()
        in paths
    )
    assert (
        Path(retention.evaluation.spatial_evaluation.place_data.__file__).resolve()
        in paths
    )


def test_substrate_receipts_bind_the_recursive_local_execution_closure() -> None:
    trainer_paths = {path.resolve() for path in scaled_trainer.training_source_paths()}
    retention_paths = {path.resolve() for path in retention.retention_source_paths()}
    required = {
        Path(scaled_trainer.spatial_model_module.__file__).resolve(),
        Path(scaled_trainer.encoder_module.__file__).resolve(),
        Path(scaled_trainer.h6_census.__file__).resolve(),
        Path(scaled_trainer.h4_v2.__file__).resolve(),
        Path(scaled_trainer.h4_v1.__file__).resolve(),
    }

    assert required <= trainer_paths
    assert trainer_paths <= retention_paths
    assert set(scaled_trainer.packer.pack_source_bindings()) == {
        "packer",
        "h6_dataset",
        "h6_main_pool_census",
        "h6_sequence_contract_v2",
        "h6_sequence_contract_v1",
    }


def test_retention_receipt_records_logical_and_physical_device_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")

    assert retention.device_receipt("cuda:0", torch.device("cuda:0")) == {
        "requested": "cuda:0",
        "resolved": "cuda:0",
        "hip_visible_devices": "1",
    }
    assert scaled_trainer.device_receipt("cuda:0", torch.device("cuda:0")) == {
        "requested": "cuda:0",
        "resolved": "cuda:0",
        "hip_visible_devices": "1",
    }


def test_diagnostic_json_writers_refuse_overwrite(tmp_path: Path) -> None:
    rank_path = tmp_path / "rank.json"
    retention_path = tmp_path / "retention.json"

    rank_probe.write_immutable_json(rank_path, {"status": "COMPLETE"})
    retention.write_immutable_json(retention_path, {"status": "COMPLETE"})

    with pytest.raises(FileExistsError):
        rank_probe.write_immutable_json(rank_path, {})
    with pytest.raises(FileExistsError):
        retention.write_immutable_json(retention_path, {})
