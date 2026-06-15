from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from lewm.benchmarks.phase2m_primitive_affordance import (
    primitive_affordance_selection_summary,
)
from lewm.benchmarks.phase2o_factorized_affordance import (
    FACTORIZED_AFFORDANCE_FACTOR_NAMES,
    build_factorized_primitive_affordance_examples,
    factorized_affordance_selection_records,
    materialize_phase2p_factorized_batch,
    phase2p_batch_contract_audit,
)
from lewm.models.primitive_affordance import (
    FactorizedPrimitiveAffordanceModel,
    factorized_affordance_losses,
    factorized_affordance_values,
)


def _image(path: Path, value: int) -> str:
    Image.new("RGB", (8, 8), color=(value, value, value)).save(path)
    return str(path)


def _labels(
    *,
    progress: float,
    p05: float = 0.2,
    minimum: float = 0.3,
    unsafe_fraction: float = 0.0,
    enters_unsafe: bool = False,
    ends_unsafe: bool = False,
    recoverable: bool = True,
    heading_error: float = 0.0,
) -> dict:
    return {
        "target_progress_m": progress,
        "p05_swept_configuration_clearance_m": p05,
        "minimum_swept_configuration_clearance_m": minimum,
        "unsafe_sample_fraction": unsafe_fraction,
        "enters_grid_unsafe": enters_unsafe,
        "ends_grid_unsafe": ends_unsafe,
        "target_recoverable": recoverable,
        "target_heading_error_rad": heading_error,
    }


def _row(
    tmp_path: Path,
    *,
    scene_id: str,
    source_index: int,
    sequence: tuple[str, str],
    progress: float,
    p05: float = 0.2,
    unsafe_fraction: float = 0.0,
    enters_unsafe: bool = False,
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
        "consequence_labels": _labels(
            progress=progress,
            p05=p05,
            unsafe_fraction=unsafe_fraction,
            enters_unsafe=enters_unsafe,
        ),
    }


def test_phase2p_materializes_factorized_source_image_batch(tmp_path: Path) -> None:
    rows = [
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("forward_slow", "hold"),
            progress=0.3,
        ),
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("backward", "hold"),
            progress=-0.3,
            p05=-0.2,
            unsafe_fraction=1.0,
            enters_unsafe=True,
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
    examples = build_factorized_primitive_affordance_examples(
        rows,
        primitive_names=("forward_slow", "backward"),
    )

    batch = materialize_phase2p_factorized_batch(examples, (0, 1), image_size=8)
    contract = phase2p_batch_contract_audit(batch)

    assert examples[0].start_frame.endswith("scene_a_1_start.png")
    assert batch.start_vision.shape == (2, 3, 8, 8)
    assert batch.primitive_utility_targets.shape == (2, 2)
    assert batch.factor_targets.shape == (
        2,
        2,
        len(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
    )
    assert batch.factor_mask.all()
    assert contract["primitive_utility_targets"] == 4
    assert contract["factor_targets"] == 4 * len(FACTORIZED_AFFORDANCE_FACTOR_NAMES)
    assert contract["all_start_frames_finite"]


def test_phase2p_factorized_model_and_loss_backpropagate() -> None:
    torch.manual_seed(97)
    model = FactorizedPrimitiveAffordanceModel(
        primitive_count=3,
        factor_count=len(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
        latent_dim=12,
        hidden_dim=24,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
    )
    logits = model(torch.randn(2, 3, 28, 28))
    losses = factorized_affordance_losses(
        factor_logits=logits,
        factor_targets=torch.rand(
            2,
            3,
            len(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
            dtype=torch.float32,
        ),
        factor_mask=torch.ones(
            2,
            3,
            len(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
            dtype=torch.bool,
        ),
        safety_weight=1.0,
        value_weight=1.0,
    )
    values = factorized_affordance_values(logits)

    losses["factorized_affordance_loss"].backward()

    assert logits.shape == (2, 3, len(FACTORIZED_AFFORDANCE_FACTOR_NAMES))
    assert values.shape == logits.shape
    assert losses["factorized_safety_valid_count"] == 6
    assert losses["factorized_value_valid_count"] == 30
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.head[-1].weight.grad is not None


def test_phase2p_safety_first_selection_uses_factor_values(tmp_path: Path) -> None:
    rows = [
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("forward_slow", "hold"),
            progress=0.3,
        ),
        _row(
            tmp_path,
            scene_id="scene_a",
            source_index=1,
            sequence=("backward", "hold"),
            progress=-0.3,
            unsafe_fraction=1.0,
            enters_unsafe=True,
        ),
        _row(
            tmp_path,
            scene_id="scene_b",
            source_index=2,
            sequence=("forward_slow", "hold"),
            progress=-0.3,
            unsafe_fraction=1.0,
            enters_unsafe=True,
        ),
        _row(
            tmp_path,
            scene_id="scene_b",
            source_index=2,
            sequence=("backward", "hold"),
            progress=0.3,
        ),
    ]
    examples = build_factorized_primitive_affordance_examples(
        rows,
        primitive_names=("forward_slow", "backward"),
    )
    factor_values = torch.tensor(
        [
            [
                [0.9, 1.0, 0.8, 0.8, 0.0, 0.5],
                [0.1, -1.0, 0.1, 0.1, 1.0, 0.5],
            ],
            [
                [0.1, -1.0, 0.1, 0.1, 1.0, 0.5],
                [0.9, 1.0, 0.8, 0.8, 0.0, 0.5],
            ],
        ],
        dtype=torch.float32,
    )

    records = factorized_affordance_selection_records(
        examples,
        factor_values,
        seed=20260615,
        split_name="validation",
        scorer_name="test_factorized",
    )
    summary = primitive_affordance_selection_summary(records)

    assert summary["primitive_match_rate"] == 1.0
    assert summary["mean_target_utility_regret"] == 0.0
    assert summary["selected_primitive_counts"] == {
        "backward": 1,
        "forward_slow": 1,
    }
    assert all(record["selected_by_predicted_safe_gate"] for record in records)
