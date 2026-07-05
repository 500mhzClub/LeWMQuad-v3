from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from lewm.benchmarks.phase2i_utility_training import (
    materialize_phase2i_utility_batch,
    phase2i_batch_contract_audit,
)
from lewm.models.source_action_utility import SourceActionUtilityRanker
from scripts.check_jepa_phase2i_utility_gate import check_gate


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
    candidate: int,
    progress: float,
    source_index: int = 1,
) -> dict:
    return {
        "scene_id": "scene_a",
        "family": "family",
        "source_index": source_index,
        "start_frame": _image(tmp_path / f"start_{candidate}.png", 20 + candidate),
        "primitive_sequence": [f"action_{candidate}", "hold"],
        "active_blocks": [[float(candidate)], [0.0]],
        "future_frames": ["future_0.png", "future_1.png"],
        "consequence_labels": _labels(progress),
    }


def test_phase2i_utility_batch_materializes_start_frames_and_groups(
    tmp_path: Path,
) -> None:
    rows = [
        _row(tmp_path, candidate=0, progress=0.3),
        _row(tmp_path, candidate=1, progress=-0.3),
    ]

    batch = materialize_phase2i_utility_batch(rows, (0, 1), image_size=8)
    audit = phase2i_batch_contract_audit(batch)

    assert batch.start_vision.shape == (2, 3, 8, 8)
    assert batch.actions.shape == (2, 2, 1)
    assert batch.action_utility_mask.tolist() == [True, True]
    assert batch.action_utility_group_ids.tolist() == [0, 0]
    assert audit["action_utility_targets"] == 2
    assert audit["action_utility_source_groups"] == 1
    assert audit["all_start_frames_finite"]


def test_source_action_utility_ranker_outputs_one_score_per_candidate() -> None:
    torch.manual_seed(67)
    model = SourceActionUtilityRanker(
        cmd_dim=2,
        horizon=2,
        latent_dim=12,
        action_hidden_dim=24,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
    )

    scores = model(torch.randn(3, 3, 28, 28), torch.randn(3, 2, 2))
    scores.sum().backward()

    assert scores.shape == (3,)
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.action_encoder[1].weight.grad is not None


def test_action_only_utility_ranker_ignores_source_image() -> None:
    torch.manual_seed(71)
    model = SourceActionUtilityRanker(
        cmd_dim=2,
        horizon=2,
        latent_dim=12,
        action_hidden_dim=24,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        input_mode="action_only",
    )
    actions = torch.randn(2, 2, 2)

    left = model(torch.randn(2, 3, 28, 28), actions)
    right = model(torch.randn(2, 3, 28, 28), actions)

    assert torch.allclose(left, right)


def test_film_interaction_ranker_supports_matched_action_only_control() -> None:
    torch.manual_seed(73)
    model = SourceActionUtilityRanker(
        cmd_dim=2,
        horizon=2,
        latent_dim=12,
        action_hidden_dim=24,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        fusion_mode="film_interaction",
    )
    action_only = SourceActionUtilityRanker(
        cmd_dim=2,
        horizon=2,
        latent_dim=12,
        action_hidden_dim=24,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        input_mode="action_only",
        fusion_mode="film_interaction",
    )
    actions = torch.randn(2, 2, 2)

    scores = model(torch.randn(2, 3, 28, 28), actions)
    left = action_only(torch.randn(2, 3, 28, 28), actions)
    right = action_only(torch.randn(2, 3, 28, 28), actions)

    assert scores.shape == (2,)
    assert model.source_conditioner is not None
    assert torch.allclose(left, right)


def test_interaction_only_action_control_cannot_rank_by_action_identity() -> None:
    torch.manual_seed(79)
    model = SourceActionUtilityRanker(
        cmd_dim=2,
        horizon=2,
        latent_dim=12,
        action_hidden_dim=24,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        input_mode="action_only",
        fusion_mode="interaction_only",
    )
    actions = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    scores = model(torch.randn(3, 3, 28, 28), actions)
    scores.sum().backward()

    assert torch.allclose(scores, scores[:1].expand_as(scores))
    assert model.action_encoder[1].weight.grad is not None
    assert torch.count_nonzero(model.action_encoder[1].weight.grad) == 0


def test_phase2i_gate_requires_absolute_and_action_only_improvement() -> None:
    passing_report = {
        "schema": "jepa_phase2i_source_action_utility_training_v0",
        "baseline_reference": [
            {
                "baseline": "full_sequence_mean",
                "top1_match_rate": 0.01,
                "first_primitive_match_rate": 0.16,
                "mean_target_utility_regret": 0.20,
            },
            {
                "baseline": "first_primitive_mean",
                "top1_match_rate": 0.02,
                "first_primitive_match_rate": 0.35,
                "mean_target_utility_regret": 0.24,
            },
        ],
        "final_validation": {
            "action_utility_selection_summary": {
                "top1_match_rate": 0.30,
                "first_primitive_match_rate": 0.55,
                "mean_target_utility_regret": 0.10,
            }
        },
    }
    failing_report = {
        **passing_report,
        "final_validation": {
            "action_utility_selection_summary": {
                "top1_match_rate": 0.20,
                "first_primitive_match_rate": 0.55,
                "mean_target_utility_regret": 0.10,
            }
        },
    }

    assert check_gate(
        passing_report,
        min_top1_match_rate=0.25,
        min_first_primitive_match_rate=0.50,
    )["passed"]
    failing = check_gate(
        failing_report,
        min_top1_match_rate=0.25,
        min_first_primitive_match_rate=0.50,
    )
    assert not failing["passed"]
    assert "top1_match_rate_below_threshold" in failing["failure_reasons"]
