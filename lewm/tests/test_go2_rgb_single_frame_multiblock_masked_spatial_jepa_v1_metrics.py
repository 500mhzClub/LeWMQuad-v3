from __future__ import annotations

from dataclasses import dataclass
import hashlib

import pytest
import torch

from lewm.benchmarks import (
    go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1 as metrics,
)
from lewm.datasets.go2_explicit_plan_discounted_successor_state_v27 import H6V2Row
from scripts.evaluate_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1 import (
    CurrentFrameH6Runtime,
)


def _independent_mask(role: str, row_index: int) -> tuple[int, ...]:
    values = []
    for quadrant, (base_row, base_column) in enumerate(
        ((0, 0), (0, 8), (8, 0), (8, 8))
    ):
        raw = (
            "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
            f"|mask|20260801|{role}|{row_index}|{quadrant}"
        ).encode("ascii")
        digest = hashlib.sha256(raw).digest()
        row_offset = int.from_bytes(digest[:4], "big") % 5
        column_offset = int.from_bytes(digest[4:8], "big") % 5
        values.extend(
            (base_row + row_offset + dr) * 16
            + base_column
            + column_offset
            + dc
            for dr in range(4)
            for dc in range(4)
        )
    return tuple(sorted(values))


def test_mask_schedule_is_exact_disjoint_and_rng_free() -> None:
    before = torch.random.get_rng_state().clone()
    for role, row_index in (("train", 0), ("train", 15_999), ("val", 2_047)):
        target, visible = metrics.mask_indices(role, row_index)
        assert target == _independent_mask(role, row_index)
        assert len(target) == 64
        assert len(visible) == 192
        assert set(target).isdisjoint(visible)
        assert sorted((*target, *visible)) == list(range(256))
        for row_base, column_base in (
            ((0, 0), (0, 8), (8, 0), (8, 8))
        ):
            block = tuple(
                index
                for index in target
                if row_base <= index // 16 < row_base + 8
                and column_base <= index % 16 < column_base + 8
            )
            rows = {index // 16 for index in block}
            columns = {index % 16 for index in block}
            assert len(block) == 16
            assert len(rows) == len(columns) == 4
            assert min(rows) >= row_base and max(rows) < row_base + 8
            assert min(columns) >= column_base and max(columns) < column_base + 8
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(metrics.MaskedSpatialMetricError):
        metrics.mask_indices("val", 2_048)


@dataclass(frozen=True)
class _MetaRow:
    index: int
    role: str
    family: str
    scene_id: str
    current_rgb: str


def test_donor_rule_uses_next_eligible_cyclic_row() -> None:
    rows = []
    for index in range(metrics.VALIDATION_ROW_COUNT):
        family = metrics.REGISTERED_FAMILIES[index % 8]
        scene = f"{family}_scene_{(index // 8) % 2}"
        rows.append(_MetaRow(index, "val", family, scene, f"rgb-{index}"))
    donors = metrics.build_validation_donor_indices(rows)
    assert len(donors) == 2_048
    for index, donor in enumerate(donors):
        assert donor == (index + 8) % 2_048
        assert rows[donor].family == rows[index].family
        assert rows[donor].scene_id != rows[index].scene_id


def test_control_summary_is_scene_then_family_equal_and_seeded() -> None:
    scenes = []
    families = []
    correct = []
    control = []
    for family_index, family in enumerate(metrics.REGISTERED_FAMILIES):
        for scene_offset in range(2):
            for _row in range(scene_offset + 1):
                scenes.append(f"{family}-{scene_offset}")
                families.append(family)
                correct.append(1.0 + family_index * 0.01)
                control.append(2.0 + family_index * 0.01)
    result = metrics.summarize_control(
        torch.tensor(correct),
        torch.tensor(control),
        scenes,
        families,
        control_name="wrong_target",
    )
    assert result.control_macro_mean - result.correct_macro_mean == pytest.approx(1.0)
    assert result.advantage_macro_mean == pytest.approx(1.0)
    assert result.advantage_bootstrap_lower_95 == pytest.approx(1.0)
    assert result.positive_family_count == 8
    assert result == metrics.summarize_control(
        torch.tensor(correct),
        torch.tensor(control),
        scenes,
        families,
        control_name="wrong_target",
    )


def test_raw_health_matches_frozen_formulas_and_detects_collapse() -> None:
    generator = torch.Generator().manual_seed(7)
    value = torch.randn(4, 256, 192, generator=generator)
    observed = metrics.raw_representation_health(value)
    position_mean = value.mean(dim=0, keepdim=True)
    expected_cross = (value - position_mean).square().mean()
    expected_spatial = (value - value.mean(dim=1, keepdim=True)).square().mean()
    assert observed.cross_sample_variance == pytest.approx(float(expected_cross), rel=1e-5)
    assert observed.within_image_spatial_diversity == pytest.approx(
        float(expected_spatial), rel=1e-5
    )
    assert 100.0 < observed.effective_rank <= 192.0

    collapsed = metrics.raw_representation_health(torch.ones(2, 256, 192))
    assert collapsed.effective_rank == 0.0
    assert collapsed.cross_sample_variance == 0.0
    assert collapsed.within_image_spatial_diversity == 0.0


@dataclass(frozen=True)
class _Reference:
    endpoint_identity_sha256: str


@dataclass(frozen=True)
class _PlaceRow:
    index: int
    role: str
    family: str
    scene_id: str
    anchor: _Reference
    positive: _Reference
    negative: _Reference


def test_flattened_place_keys_retrieve_positives_and_report_exact_chance() -> None:
    rows = []
    for family in metrics.REGISTERED_FAMILIES:
        for _ in range(metrics.PLACE_FAMILY_ROW_COUNTS[family]):
            index = len(rows)
            rows.append(
                _PlaceRow(
                    index=index,
                    role="checkpoint_selection",
                    family=family,
                    scene_id=f"{family}-scene",
                    anchor=_Reference(f"a-{index}"),
                    positive=_Reference(f"p-{index}"),
                    negative=_Reference(f"n-{index}"),
                )
            )
    positive = torch.eye(320)
    online = positive.clone()
    negative = positive.roll(1, dims=0)
    anchor = positive.roll(2, dims=0)
    result = metrics.evaluate_place_keys(
        rows, online, anchor, positive, negative
    )
    assert result["retrieval"]["recall_at_5"] == 1.0
    # Candidate counts are 64 for seven scenes and 60 for one.
    assert result["retrieval"]["chance_multiple"] == pytest.approx(12.7)
    assert result["retrieval"]["chance_multiple"] != pytest.approx(
        result["retrieval"]["aggregate_recall_over_aggregate_chance"]
    )
    assert result["retrieval"]["scene_count_at_least_1_5x_chance"] == 8
    assert result["energy"]["negative_minus_positive_macro_mean"] > 0.0
    assert result["energy"]["negative_minus_positive_bootstrap_lower_95"] > 0.0
    assert result["target_place_key_effective_rank"] > 300.0


class _CurrentOnlyLoader:
    def __init__(self) -> None:
        self.indices: list[tuple[str, int]] = []

    def load_current(self, row: H6V2Row) -> torch.Tensor:
        self.indices.append((row.role, row.index))
        return torch.full((3, 112, 112), float(row.index % 17))

    def access_snapshot(self) -> dict[str, int]:
        return {"current": len(self.indices)}


def _h6_rows(role: str, count: int, prefix: str) -> tuple[H6V2Row, ...]:
    rows = []
    for index in range(count):
        family = metrics.REGISTERED_FAMILIES[index % 8]
        scene = f"{prefix}-{family}-{(index // 8) % 8}"
        rgb = tuple(f"{scene}/rgb/frame-{index}-{offset}.png" for offset in range(7))
        rows.append(H6V2Row(index, role, family, scene, rgb, (0, 1, 2, 3, 4, 5)))
    return tuple(rows)


def test_current_frame_runtime_returns_exact_four_b4_batches_only() -> None:
    loader = _CurrentOnlyLoader()
    runtime = CurrentFrameH6Runtime(
        _h6_rows("train", 16_000, "train"),
        _h6_rows("val", 2_048, "val"),
        loader=loader,  # type: ignore[arg-type]
        device="cpu",
    )
    batches = runtime.train_rows_for_update(2)
    assert [batch.row_indices for batch in batches] == [
        (16, 17, 18, 19),
        (20, 21, 22, 23),
        (24, 25, 26, 27),
        (28, 29, 30, 31),
    ]
    assert all(tuple(batch.rgb.shape) == (4, 3, 112, 112) for batch in batches)
    assert loader.indices == [("train", index) for index in range(16, 32)]
