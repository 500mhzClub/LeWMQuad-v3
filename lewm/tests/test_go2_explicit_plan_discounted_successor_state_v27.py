from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import FAMILIES
from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as v27


ROOT = Path(__file__).resolve().parents[2]


def _row_value(
    *,
    family: str,
    scene_suffix: str,
    env_index: int = 0,
    actions: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
) -> dict[str, object]:
    scene_id = f"{family}_{scene_suffix}"
    return {
        "schema": v27.CORRECTED_H6_V2_ROW_SCHEMA,
        "role": "val",
        "family": family,
        "scene_id": scene_id,
        "rgb": [
            f"{scene_id}/rgb/frame_{env_index + 240 * offset:06d}_env_{env_index:02d}.png"
            for offset in range(7)
        ],
        "actions": list(actions),
    }


def _metric_row(index: int, family: str, scene_number: int) -> v27.H6V2Row:
    scene_id = f"{family}_{scene_number:012x}"
    env_index = index % 48
    return v27.H6V2Row(
        index=index,
        role="val",
        family=family,
        scene_id=scene_id,
        rgb=tuple(
            f"{scene_id}/rgb/frame_{env_index + 240 * offset:06d}_env_{env_index:02d}.png"
            for offset in range(7)
        ),
        actions=(0, 1, 2, 3, 4, 5),
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def test_exact_frozen_index_and_image_constants() -> None:
    assert v27.INDEX_BINDINGS["train"].path == Path(
        ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/train.jsonl"
    )
    assert v27.INDEX_BINDINGS["train"].row_count == 16_000
    assert v27.INDEX_BINDINGS["train"].byte_count == 10_328_000
    assert v27.INDEX_BINDINGS["train"].sha256 == (
        "aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77"
    )
    assert v27.INDEX_BINDINGS["val"].row_count == 2_048
    assert v27.INDEX_BINDINGS["val"].byte_count == 1_317_888
    assert v27.INDEX_BINDINGS["val"].sha256 == (
        "83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6"
    )
    assert v27.TRAIN_PREFIX_ROWS == 6_400
    assert v27.CROP_BOX == (0, 28, 224, 196)
    assert v27.MODEL_IMAGE_SIZE == (112, 112)
    assert v27.PLAN_GAMMA == 0.9
    assert v27.PLAN_WEIGHTS == (1.0, 0.9, 0.81, 0.729)
    assert v27.PLAN_WEIGHT_SUM == 3.439


def test_strict_corrected_h6_row_parser_and_causal_endpoints() -> None:
    value = _row_value(
        family=FAMILIES[0],
        scene_suffix="0123456789ab",
        env_index=17,
    )
    row = v27._decode_row(value, role="val", index=4)
    assert row.index == 4
    assert row.current_rgb == value["rgb"][2]
    assert row.future_rgb == tuple(value["rgb"][3:7])
    assert row.plan == (2, 3, 4, 5)
    assert row.first_plan_action == 2

    bad = json.loads(_canonical(value))
    bad["rgb"][4] = bad["rgb"][4].replace("000977", "000978")
    with pytest.raises(v27.V27DataContractError, match="numeric identity|causal"):
        v27._decode_row(bad, role="val", index=4)

    bad = json.loads(_canonical(value))
    bad["actions"][2] = True
    with pytest.raises(v27.V27DataContractError, match="six action IDs"):
        v27._decode_row(bad, role="val", index=4)

    bad = json.loads(_canonical(value))
    bad["rgb"][0] = "../rgb/frame_000017_env_17.png"
    with pytest.raises(v27.V27DataContractError, match="allowlist"):
        v27._decode_row(bad, role="val", index=4)

    with pytest.raises(v27.V27DataContractError, match="duplicate JSON key"):
        v27._strict_json_loads(b'{"actions":[],"actions":[]}')


def test_frozen_donor_selection_uses_positive_cyclic_offset() -> None:
    assert v27.select_donor_index(
        row_index=3,
        candidate_indices=(0, 4, 7),
        predicate=lambda _index: True,
        modulus=8,
    ) == 4
    assert v27.select_donor_index(
        row_index=3,
        candidate_indices=(0, 4, 7),
        predicate=lambda index: index != 4,
        modulus=8,
    ) == 7
    assert v27.select_donor_index(
        row_index=3,
        candidate_indices=(3,),
        predicate=lambda _index: True,
        modulus=8,
    ) is None


def test_rgb_rectification_matches_exact_pillow_crop_resize_and_normalization() -> None:
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")

    y, x = np.indices((224, 224))
    pixels = np.stack(
        ((x + y) % 256, (2 * x + y) % 256, (x + 3 * y) % 256), axis=-1
    ).astype(np.uint8)
    source = Image.fromarray(pixels, mode="RGB")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    observed = v27.rectify_h6_rgb_bytes(buffer.getvalue())
    expected_image = source.crop((0, 28, 224, 196)).resize(
        (112, 112), Image.Resampling.BILINEAR
    )
    expected = torch.from_numpy(np.array(expected_image, copy=True)).permute(2, 0, 1)
    expected = expected.contiguous().to(dtype=torch.float32).div_(255.0)
    expected.sub_(
        torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    ).div_(
        torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    )
    assert observed.dtype == torch.float32
    assert tuple(observed.shape) == (3, 112, 112)
    assert torch.equal(observed, expected)

    wrong_size = io.BytesIO()
    Image.new("RGB", (224, 223)).save(wrong_size, format="PNG")
    with pytest.raises(v27.V27DataContractError, match="224x224 RGB PNG"):
        v27.rectify_h6_rgb_bytes(wrong_size.getvalue())

    wrong_mode = io.BytesIO()
    Image.new("RGBA", (224, 224)).save(wrong_mode, format="PNG")
    with pytest.raises(v27.V27DataContractError, match="224x224 RGB PNG"):
        v27.rectify_h6_rgb_bytes(wrong_mode.getvalue())


def test_discounted_successor_target_is_float32_and_stop_gradient() -> None:
    torch = pytest.importorskip("torch")
    future = torch.stack(
        [torch.full((64, 64, 64), float(value)) for value in (1, 2, 3, 4)]
    )[None].requires_grad_(True)
    target = v27.discounted_successor_target(future)
    expected = (1.0 + 0.9 * 2.0 + 0.81 * 3.0 + 0.729 * 4.0) / 3.439
    assert tuple(target.shape) == (1, 64, 64, 64)
    assert target.dtype == torch.float32
    assert target.requires_grad is False
    assert torch.allclose(target, torch.full_like(target, expected), atol=1e-6, rtol=0.0)

    with pytest.raises(ValueError, match="shape"):
        v27.discounted_successor_target(future[:, :3])
    with pytest.raises(v27.V27DataContractError, match="float32"):
        v27.discounted_successor_target(future.double())


def _bootstrap_rows() -> tuple[list[v27.H6V2Row], list[float]]:
    rows: list[v27.H6V2Row] = []
    values: list[float] = []
    for family_index, family in enumerate(reversed(v27.LEXICOGRAPHIC_FAMILIES)):
        for scene_index in range(3):
            rows.append(_metric_row(len(rows), family, 100 + scene_index))
            values.append(float(family_index + scene_index) / 10.0)
    return rows, values


def test_scene_family_aggregation_and_fresh_pcg64_draw_order() -> None:
    rows, values = _bootstrap_rows()
    observed = v27.aggregate_normalized_advantage(
        rows,
        values,
        observation_update=400,
        metric_name="tail_advantage",
    )

    by_family: dict[str, dict[str, float]] = {
        family: {} for family in v27.LEXICOGRAPHIC_FAMILIES
    }
    for row, value in zip(rows, values, strict=True):
        by_family[row.family][row.scene_id] = value
    rng = np.random.Generator(np.random.PCG64(20_260_730))
    family_replicates = []
    for family in v27.LEXICOGRAPHIC_FAMILIES:
        vector = np.asarray(
            [by_family[family][scene] for scene in sorted(by_family[family])],
            dtype=np.float64,
        )
        indices = rng.integers(0, len(vector), size=(2_000, len(vector)))
        family_replicates.append(vector[indices].mean(axis=1))
    expected_lower = float(
        np.sort(np.stack(family_replicates, axis=0).mean(axis=0))[50]
    )
    assert observed["bootstrap_lower_95"] == expected_lower
    assert observed["positive_family_count"] == 8

    fresh_identity = v27.aggregate_normalized_advantage(
        rows,
        values,
        observation_update=100,
        metric_name="wrong_plan_advantage",
    )
    assert fresh_identity["bootstrap_lower_95"] == expected_lower


def test_rows_are_averaged_before_scenes_and_families() -> None:
    rows: list[v27.H6V2Row] = []
    values: dict[int, float] = {}
    for family_index, family in enumerate(v27.LEXICOGRAPHIC_FAMILIES):
        first = _metric_row(len(rows), family, 1)
        rows.append(first)
        values[first.index] = 1.0
        if family_index == 0:
            repeated = _metric_row(len(rows), family, 1)
            rows.append(repeated)
            values[repeated.index] = 3.0
            second = _metric_row(len(rows), family, 2)
            rows.append(second)
            values[second.index] = 6.0
    result = v27.aggregate_normalized_advantage(
        rows,
        values,
        observation_update=400,
        metric_name="persistence_advantage",
    )
    assert result["family_equal_scene_means"][v27.LEXICOGRAPHIC_FAMILIES[0]] == 4.0
    assert result["equal_family_mean"] == pytest.approx((4.0 + 7.0) / 8.0)


def test_plan_energy_summary_uses_unclamped_ratio_and_per_row_denominators() -> None:
    rows = [
        _metric_row(index, family, index + 1)
        for index, family in enumerate(v27.LEXICOGRAPHIC_FAMILIES)
    ]
    count = len(rows)
    result = v27.summarize_plan_energies(
        rows,
        observation_update=400,
        correct_energy=[0.5] * count,
        persistence_energy=[1.0] * count,
        wrong_plan_energy=[0.75] * count,
        tail_energy=[0.60] * count,
        wrong_scene_energy={row.index: 0.90 for row in rows},
        mean_prior_energy=[0.80] * count,
    )
    assert result["correct_ratio"] == 0.5
    assert result["advantages"]["persistence_advantage"]["equal_family_mean"] == 0.5
    assert result["advantages"]["wrong_plan_advantage"]["equal_family_mean"] == 0.25
    assert result["advantages"]["tail_advantage"]["equal_family_mean"] == pytest.approx(0.10)
    assert result["advantages"]["wrong_scene_advantage"]["equal_family_mean"] == pytest.approx(0.40)
    assert result["advantages"]["mean_prior_advantage"]["equal_family_mean"] == pytest.approx(0.30)


def test_real_bound_metadata_preflight_reproduces_frozen_donor_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_rgb_decode(_raw: bytes) -> object:
        raise AssertionError("metadata preflight attempted RGB decoding")

    monkeypatch.setattr(v27, "rectify_h6_rgb_bytes", forbidden_rgb_decode)
    result = v27.metadata_only_preflight(ROOT)
    assert result["status"] == "PASS_METADATA_ONLY_PREFLIGHT"
    assert result["train"]["row_count"] == 16_000
    assert result["validation"]["row_count"] == 2_048
    assert result["donors"] == {
        "rule": v27.DONOR_RULE,
        "modulus": 2_048,
        "tail_donor_count": 2_048,
        "wrong_plan_donor_count": 2_048,
        "exact_plan_wrong_scene_row_count": 1_212,
        "exact_plan_counts_by_family": {
            "large_enclosed_maze": 137,
            "local_composite_motifs": 144,
            "loop_alias_stress": 141,
            "medium_enclosed_maze": 159,
            "open_obstacle_field": 184,
            "rough_local_dynamics": 170,
            "small_enclosed_maze": 127,
            "visual_sensor_stress": 150,
        },
        "panel_sha256": "f6771ae89cd5dd32338d75516a6b822b40842da9260c28d236905209f372286e",
    }
    assert result["rgb_open_count"] == 0
    assert result["gpu_use_count"] == 0
    assert result["generated_write_count"] == 0
