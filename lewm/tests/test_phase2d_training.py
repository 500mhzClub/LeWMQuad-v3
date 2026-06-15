from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from lewm.benchmarks.phase2_data import build_hard_negative_index
from lewm.benchmarks.phase2d_training import (
    ACTION_UTILITY_TARGET_VERSION,
    CONSEQUENCE_TARGET_DIM,
    action_utility_selection_records,
    action_utility_selection_summary,
    action_utility_target,
    batch_contract_audit,
    checkpoint_rule_record,
    consequence_target_vector,
    materialize_phase2d_batch,
    prediction_control_records,
    primary_source_state_prediction_table,
    registered_cell,
)
from scripts.train_jepa_phase2d import _assert_finite_metrics


def _image(path: Path, value: int) -> str:
    Image.new("RGB", (8, 8), color=(value, value, value)).save(path)
    return str(path)


def _row(
    tmp_path: Path,
    *,
    candidate: int,
    actions: list[list[float]],
    valid: list[bool],
) -> dict:
    start = _image(tmp_path / f"start_{candidate}.png", 20)
    future = [
        _image(tmp_path / f"future_{candidate}_{step}.png", 40 + step)
        if is_valid
        else None
        for step, is_valid in enumerate(valid)
    ]
    return {
        "scene_id": "scene_a",
        "family": "family",
        "source_index": 7,
        "start_frame": start,
        "primitive_sequence": [f"action_{value[0]}" for value in actions],
        "active_blocks": actions,
        "future_frames": future,
        "future_observations": [
            {"rgb_path": path, "observation_valid": is_valid}
            for path, is_valid in zip(future, valid, strict=True)
        ],
    }


def test_registered_cells_fix_model_side_factorial_differences() -> None:
    assert registered_cell("C0").target_ema_momentum is None
    assert registered_cell("C1").target_ema_momentum == 0.99
    assert registered_cell("C2").action_identifiability_lambda == 1.0
    assert registered_cell("state_only").prediction_input_mode == "state_only"
    assert not registered_cell("action_only").participates_in_checkpoint_selection
    with pytest.raises(ValueError, match="unknown"):
        registered_cell("unregistered")


def test_phase2d_trainer_rejects_nonfinite_metrics() -> None:
    with pytest.raises(RuntimeError, match="nonfinite_phase2d_metrics"):
        _assert_finite_metrics(
            {"loss": float("nan"), "prediction_loss": 0.0},
            step=3,
            phase="train",
        )


def test_materialized_phase2d_batch_emits_masks_and_exhaustive_negatives(
    tmp_path: Path,
) -> None:
    rows = [
        _row(tmp_path, candidate=0, actions=[[1.0], [2.0]], valid=[True, False]),
        _row(tmp_path, candidate=1, actions=[[2.0], [1.0]], valid=[True, True]),
        _row(tmp_path, candidate=2, actions=[[0.0], [0.0]], valid=[True, True]),
    ]
    negatives = [
        build_hard_negative_index(rows, step=step) for step in range(2)
    ]

    batch = materialize_phase2d_batch(
        rows,
        (0, 1, 2),
        hard_negatives=negatives,
        image_size=8,
    )
    audit = batch_contract_audit(batch)

    assert batch.vision.shape == (3, 3, 3, 8, 8)
    assert batch.actions.shape == (3, 2, 1)
    assert batch.transition_mask.tolist() == [
        [True, False],
        [True, True],
        [True, True],
    ]
    assert batch.non_hold_mask.tolist() == [
        [True, True],
        [True, True],
        [False, False],
    ]
    assert batch.wrong_mask[0, 0].sum() == 2
    assert batch.wrong_mask[1, 0].sum() == 2
    assert batch.wrong_mask[2].sum() == 0
    assert torch.allclose(batch.vision[0, 1], batch.vision[0, 2])
    assert audit["valid_transitions"] == 5
    assert audit["eligible_wrong_pairs"] == 6
    assert batch.consequence_targets.shape == (3, CONSEQUENCE_TARGET_DIM)
    assert not batch.consequence_mask.any()
    assert audit["consequence_label_fields"] == CONSEQUENCE_TARGET_DIM
    assert audit["consequence_label_values"] == 0
    assert batch.action_utility_targets.shape == (3,)
    assert not batch.action_utility_mask.any()
    assert batch.action_utility_group_ids.tolist() == [0, 0, 0]
    assert audit["action_utility_targets"] == 0
    assert audit["action_utility_source_groups"] == 1
    assert audit["action_utility_target_version"] == ACTION_UTILITY_TARGET_VERSION
    assert audit["all_materialized_frames_finite"]


def test_consequence_target_vector_normalizes_and_masks_nullable_labels() -> None:
    values, mask = consequence_target_vector(
        {
            "consequence_labels": {
                "target_progress_m": 0.15,
                "clearance_gain_m": -0.6,
                "minimum_swept_configuration_clearance_m": 0.4,
                "p05_swept_configuration_clearance_m": -0.2,
                "unsafe_sample_fraction": 0.25,
                "enters_grid_unsafe": True,
                "ends_grid_unsafe": False,
                "target_recoverable": True,
                "target_heading_error_rad": None,
            }
        }
    )

    assert len(values) == CONSEQUENCE_TARGET_DIM
    assert len(mask) == CONSEQUENCE_TARGET_DIM
    assert values[:8] == pytest.approx(
        (
            0.5,
            -1.0,
            0.5,
            0.0,
            0.25,
            1.0,
            0.0,
            1.0,
        )
    )
    assert values[8] == 0.0
    assert mask == (True, True, True, True, True, True, True, True, False)


def _utility_labels(
    *,
    progress: float | None = 0.0,
    clearance_gain: float | None = None,
    p05: float | None = 0.2,
    unsafe_fraction: float | None = 0.0,
    enters_unsafe: bool = False,
    ends_unsafe: bool = False,
    recoverable: bool = True,
    heading_error: float | None = 0.0,
) -> dict:
    labels = {
        "p05_swept_configuration_clearance_m": p05,
        "unsafe_sample_fraction": unsafe_fraction,
        "enters_grid_unsafe": enters_unsafe,
        "ends_grid_unsafe": ends_unsafe,
        "target_recoverable": recoverable,
        "target_heading_error_rad": heading_error,
    }
    if progress is not None:
        labels["target_progress_m"] = progress
    if clearance_gain is not None:
        labels["clearance_gain_m"] = clearance_gain
    return labels


def test_action_utility_target_is_safety_first_and_masked() -> None:
    safe_value, safe_mask = action_utility_target(
        {"consequence_labels": _utility_labels(progress=0.05)}
    )
    unsafe_value, unsafe_mask = action_utility_target(
        {
            "consequence_labels": _utility_labels(
                progress=0.3,
                p05=1.0,
                enters_unsafe=True,
                unsafe_fraction=1.0,
            )
        }
    )
    missing_value, missing_mask = action_utility_target(
        {"consequence_labels": {"target_progress_m": 0.1}}
    )

    assert safe_mask
    assert unsafe_mask
    assert safe_value > unsafe_value
    assert missing_value == 0.0
    assert not missing_mask


def test_materialized_phase2d_batch_caches_duplicate_image_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_start = _image(tmp_path / "shared_start.png", 20)
    rows = [
        _row(tmp_path, candidate=0, actions=[[1.0]], valid=[False]),
        _row(tmp_path, candidate=1, actions=[[2.0]], valid=[False]),
    ]
    for row in rows:
        row["start_frame"] = shared_start
        row["future_frames"] = [None]
        row["future_observations"] = [
            {"rgb_path": None, "observation_valid": False}
        ]
    negatives = [build_hard_negative_index(rows, step=0)]
    calls = []

    def fake_image_tensor(path: Path, *, image_size: int = 224) -> torch.Tensor:
        calls.append(path)
        return torch.zeros(3, image_size, image_size)

    monkeypatch.setattr(
        "lewm.benchmarks.phase2d_training.image_tensor",
        fake_image_tensor,
    )

    batch = materialize_phase2d_batch(
        rows,
        (0, 1),
        hard_negatives=negatives,
        image_size=8,
    )

    assert batch.vision.shape == (2, 2, 3, 8, 8)
    assert calls == [Path(shared_start)]


def test_materialized_phase2d_batch_rejects_split_source_groups(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, candidate=0, actions=[[1.0]], valid=[True]),
        _row(tmp_path, candidate=1, actions=[[2.0]], valid=[True]),
    ]
    negatives = [build_hard_negative_index(rows, step=0)]

    with pytest.raises(ValueError, match="omitted registered hard negatives"):
        materialize_phase2d_batch(
            rows,
            (0,),
            hard_negatives=negatives,
            image_size=8,
        )


def test_prediction_control_records_and_source_state_table(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, candidate=0, actions=[[1.0], [2.0]], valid=[True, False]),
        _row(tmp_path, candidate=1, actions=[[2.0], [1.0]], valid=[True, True]),
        _row(tmp_path, candidate=2, actions=[[0.0], [0.0]], valid=[True, True]),
    ]
    batch = materialize_phase2d_batch(
        rows,
        (0, 1, 2),
        hard_negatives=[build_hard_negative_index(rows, step=step) for step in range(2)],
        image_size=8,
    )
    output = {
        "real_mse": torch.tensor([[1.0, 9.0], [2.0, 3.0], [4.0, 5.0]]),
        "mean_wrong_mse": torch.tensor([[3.0, 0.0], [5.0, 7.0], [0.0, 0.0]]),
        "zero_mse": torch.tensor([[2.0, 0.0], [6.0, 8.0], [0.0, 0.0]]),
        "target_change_mse": torch.tensor([[2.0, 1.0], [4.0, 6.0], [8.0, 10.0]]),
        "eligible_wrong_mask": batch.wrong_mask.any(dim=2),
        "eligible_zero_mask": batch.non_hold_mask & batch.transition_mask,
        "transition_mask": batch.transition_mask,
    }

    candidate_records = prediction_control_records(batch, output, seed=20260614)
    source_records = primary_source_state_prediction_table(candidate_records)
    checkpoint = checkpoint_rule_record(
        source_records,
        epoch=3,
        stability={
            "collapse_warning": False,
            "effective_rank_warning": False,
            "near_static_target_warning": False,
            "mean_feature_std": 0.2,
        },
    )

    assert len(candidate_records) == 5
    first = candidate_records[0]
    assert first["step"] == 1
    assert first["hard_negative_action_advantage"] == 2.0
    assert first["hard_negative_action_advantage_over_target_change"] == 1.0
    assert first["zero_action_advantage_over_target_change"] == 0.5
    assert len(source_records) == 1
    assert source_records[0]["candidate_rows"] == 3
    assert source_records[0]["eligible_wrong_candidate_rows"] == 2
    assert source_records[0]["eligible_zero_candidate_rows"] == 2
    assert source_records[0]["one_step_rollout_persistence_ratio"] == pytest.approx(
        (1.0 + 2.0 + 4.0) / (2.0 + 4.0 + 8.0)
    )
    assert checkpoint["epoch"] == 3
    assert checkpoint["stability_pass"]
    assert checkpoint["hard_negative_action_advantage_pass"]
    assert checkpoint["zero_action_advantage_pass"]
    assert checkpoint["persistence_pass"]
    assert checkpoint["gate_pass"]
    assert checkpoint["source_state_count"] == 1


def test_action_utility_selection_records_are_source_local(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, candidate=0, actions=[[1.0]], valid=[True]),
        _row(tmp_path, candidate=1, actions=[[2.0]], valid=[True]),
        _row(tmp_path, candidate=2, actions=[[3.0]], valid=[True]),
    ]
    rows[0]["consequence_labels"] = _utility_labels(progress=0.0)
    rows[1]["consequence_labels"] = _utility_labels(progress=0.3)
    rows[2]["consequence_labels"] = _utility_labels(progress=0.1)
    batch = materialize_phase2d_batch(
        rows,
        (0, 1, 2),
        hard_negatives=[build_hard_negative_index(rows, step=0)],
        image_size=8,
    )
    output = {
        "action_utility_prediction": torch.tensor([0.2, 0.1, 0.9]),
    }

    records = action_utility_selection_records(batch, output, seed=20260614)
    summary = action_utility_selection_summary(records)

    assert len(records) == 1
    record = records[0]
    assert record["candidate_rows"] == 3
    assert record["selected_row_index"] == 2
    assert record["oracle_row_index"] == 1
    assert record["top1_match"] is False
    assert record["target_utility_regret"] > 0.0
    assert summary is not None
    assert summary["source_state_count"] == 1
    assert summary["top1_match_rate"] == 0.0
    assert summary["mean_candidate_rows"] == 3.0


def test_checkpoint_rule_blocks_scientifically_invalid_validation() -> None:
    checkpoint = checkpoint_rule_record(
        [
            {
                "hard_negative_action_advantage_over_target_change": 0.2,
                "zero_action_advantage_over_target_change": 0.2,
                "one_step_rollout_persistence_ratio": 0.8,
            },
            {
                "hard_negative_action_advantage_over_target_change": -0.1,
                "zero_action_advantage_over_target_change": -0.1,
                "one_step_rollout_persistence_ratio": 2.0,
            },
        ],
        epoch=1,
        stability={
            "collapse_warning": True,
            "effective_rank_warning": False,
            "near_static_target_warning": False,
            "mean_feature_std": 0.02,
        },
    )

    assert not checkpoint["stability_pass"]
    assert not checkpoint["hard_negative_action_advantage_pass"]
    assert not checkpoint["zero_action_advantage_pass"]
    assert not checkpoint["persistence_pass"]
    assert not checkpoint["gate_pass"]
