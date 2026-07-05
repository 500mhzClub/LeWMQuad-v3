from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from lewm.benchmarks.phase2m_primitive_affordance import (
    build_primitive_affordance_examples,
    evaluate_primitive_action_only_baseline,
    fit_primitive_action_priors,
    materialize_phase2m_primitive_batch,
    oracle_primitive_class_weights,
    phase2m_batch_contract_audit,
    primitive_affordance_dataset_audit,
    primitive_affordance_selection_records,
    primitive_affordance_selection_summary,
)
from lewm.models.primitive_affordance import (
    PrimitiveAffordanceModel,
    primitive_affordance_losses,
)
from scripts.check_jepa_phase2m_primitive_affordance_gate import check_gate


def _image(path: Path, value: int) -> str:
    Image.new("RGB", (8, 8), color=(value, value, value)).save(path)
    return str(path)


def _labels(progress: float) -> dict:
    return {
        "target_progress_m": progress,
        "p05_swept_configuration_clearance_m": 0.2,
        "unsafe_sample_fraction": 0.0,
        "enters_grid_unsafe": False,
        "ends_grid_unsafe": False,
        "target_recoverable": True,
        "target_heading_error_rad": 0.0,
    }


def _row(
    tmp_path: Path,
    *,
    scene_id: str,
    source_index: int,
    sequence: tuple[str, str],
    progress: float,
) -> dict:
    start = _image(
        tmp_path / f"{scene_id}_{source_index}_start.png",
        20 + source_index,
    )
    return {
        "scene_id": scene_id,
        "family": "family",
        "source_index": source_index,
        "start_frame": start,
        "primitive_sequence": list(sequence),
        "active_blocks": [[1.0], [2.0]],
        "future_frames": ["future_0.png", "future_1.png"],
        "consequence_labels": _labels(progress),
    }


def test_phase2m_primitive_labels_take_best_continuation_per_first_primitive(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("forward_slow", "hold"),
            progress=0.0,
        ),
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("forward_slow", "yaw_left"),
            progress=0.3,
        ),
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("backward", "hold"),
            progress=-0.3,
        ),
    ]

    examples = build_primitive_affordance_examples(
        rows,
        primitive_names=("forward_slow", "backward"),
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.valid_primitive_count == 2
    assert example.oracle_primitive == "forward_slow"
    assert example.oracle_sequence == ("forward_slow", "yaw_left")
    assert example.utility_targets[0] > example.utility_targets[1]


def test_phase2m_batch_and_dataset_audit_materialize_source_images(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("forward_slow", "hold"),
            progress=0.2,
        ),
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("backward", "hold"),
            progress=-0.2,
        ),
        _row(
            tmp_path,
            scene_id="scene_b",
            source_index=2,
            sequence=("forward_slow", "hold"),
            progress=-0.1,
        ),
        _row(
            tmp_path,
            scene_id="scene_b",
            source_index=2,
            sequence=("backward", "hold"),
            progress=0.1,
        ),
    ]
    examples = build_primitive_affordance_examples(
        rows,
        primitive_names=("forward_slow", "backward"),
    )

    batch = materialize_phase2m_primitive_batch(examples, (0, 1), image_size=8)
    contract = phase2m_batch_contract_audit(batch)
    audit = primitive_affordance_dataset_audit(examples, split_name="train")

    assert batch.start_vision.shape == (2, 3, 8, 8)
    assert batch.primitive_utility_targets.shape == (2, 2)
    assert batch.primitive_utility_mask.tolist() == [[True, True], [True, True]]
    assert contract["primitive_utility_targets"] == 4
    assert contract["source_states_with_two_or_more_valid_primitives"] == 2
    assert audit["source_states"] == 2
    assert audit["oracle_primitive_counts"] == {
        "backward": 1,
        "forward_slow": 1,
    }


def test_primitive_affordance_model_and_loss_backpropagate() -> None:
    torch.manual_seed(83)
    model = PrimitiveAffordanceModel(
        primitive_count=3,
        latent_dim=12,
        hidden_dim=24,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
    )
    scores = model(torch.randn(2, 3, 28, 28))
    losses = primitive_affordance_losses(
        primitive_scores=scores,
        primitive_utility_targets=torch.tensor(
            [[0.1, 0.3, -0.2], [0.0, 0.2, 0.1]],
            dtype=torch.float32,
        ),
        primitive_utility_mask=torch.tensor(
            [[True, True, True], [True, True, False]],
            dtype=torch.bool,
        ),
        regression_weight=1.0,
        ranking_loss="soft_ce",
        softmax_temperature=0.25,
    )

    losses["primitive_affordance_loss"].backward()

    assert scores.shape == (2, 3)
    assert losses["primitive_affordance_valid_count"] == 5
    assert losses["primitive_affordance_source_count"] == 2
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.head[-1].weight.grad is not None


def test_phase2n_oracle_inverse_frequency_weights_emphasize_rare_primitives(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            tmp_path,
            scene_id=f"scene_{index}",
            source_index=index,
            sequence=("forward_slow", "hold"),
            progress=0.3 if index < 3 else -0.3,
        )
        for index in range(4)
    ]
    rows.extend(
        [
            _row(
                tmp_path,
                scene_id=f"scene_{index}",
                source_index=index,
                sequence=("backward", "hold"),
                progress=-0.3 if index < 3 else 0.3,
            )
            for index in range(4)
        ]
    )
    examples = build_primitive_affordance_examples(
        rows,
        primitive_names=("forward_slow", "backward"),
    )

    weights = oracle_primitive_class_weights(examples)
    scores = torch.zeros(2, 2, requires_grad=True)
    weighted = primitive_affordance_losses(
        primitive_scores=scores,
        primitive_utility_targets=torch.tensor(
            [[0.3, -0.3], [-0.3, 0.3]],
            dtype=torch.float32,
        ),
        primitive_utility_mask=torch.ones(2, 2, dtype=torch.bool),
        primitive_class_weights=torch.tensor(weights, dtype=torch.float32),
        regression_weight=0.0,
        ranking_loss="hard_ce",
    )

    assert weights[1] > weights[0]
    assert weighted["primitive_affordance_source_count"] == 2
    weighted["primitive_affordance_loss"].backward()
    assert scores.grad is not None


def test_phase2m_selection_summary_and_action_only_prior(
    tmp_path: Path,
) -> None:
    rows = [
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("forward_slow", "hold"),
            progress=0.2,
        ),
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("backward", "hold"),
            progress=-0.2,
        ),
        _row(
            tmp_path,
            scene_id="scene_b",
            source_index=2,
            sequence=("forward_slow", "hold"),
            progress=-0.2,
        ),
        _row(
            tmp_path,
            scene_id="scene_b",
            source_index=2,
            sequence=("backward", "hold"),
            progress=0.2,
        ),
    ]
    examples = build_primitive_affordance_examples(
        rows,
        primitive_names=("forward_slow", "backward"),
    )
    priors = fit_primitive_action_priors(examples)
    baseline = evaluate_primitive_action_only_baseline(
        examples,
        priors,
        split_name="validation",
        seed=20260615,
    )
    model_records = primitive_affordance_selection_records(
        examples,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        seed=20260615,
        split_name="validation",
        scorer_name="test_model",
    )
    summary = primitive_affordance_selection_summary(model_records)

    assert baseline["selection_summary"]["source_state_count"] == 2
    assert summary["primitive_match_rate"] == 1.0
    assert summary["mean_target_utility_regret"] == 0.0
    assert summary["selected_primitive_counts"] == {
        "backward": 1,
        "forward_slow": 1,
    }


def test_phase2m_gate_requires_accuracy_regret_and_noncollapsed_selection() -> None:
    passing_report = {
        "schema": "jepa_phase2m_primitive_affordance_training_v0",
        "primitive_action_only_baseline": {
            "selection_summary": {
                "primitive_match_rate": 0.40,
                "mean_target_utility_regret": 0.20,
                "selected_max_primitive_fraction": 1.0,
            }
        },
        "final_validation": {
            "primitive_affordance_selection_summary": {
                "primitive_match_rate": 0.60,
                "mean_target_utility_regret": 0.10,
                "selected_max_primitive_fraction": 0.45,
                "oracle_max_primitive_fraction": 0.35,
            }
        },
    }
    failing_report = {
        **passing_report,
        "final_validation": {
            "primitive_affordance_selection_summary": {
                "primitive_match_rate": 0.45,
                "mean_target_utility_regret": 0.25,
                "selected_max_primitive_fraction": 0.80,
                "oracle_max_primitive_fraction": 0.35,
            }
        },
    }

    assert check_gate(
        passing_report,
        min_primitive_match_rate=0.50,
        max_selected_primitive_excess=0.20,
    )["passed"]
    failing = check_gate(
        failing_report,
        min_primitive_match_rate=0.50,
        max_selected_primitive_excess=0.20,
    )
    assert not failing["passed"]
    assert "primitive_match_rate_below_threshold" in failing["failure_reasons"]
    assert "regret_not_below_action_only_baseline" in failing["failure_reasons"]
    assert (
        "selected_primitive_distribution_more_collapsed_than_oracle"
        in failing["failure_reasons"]
    )
