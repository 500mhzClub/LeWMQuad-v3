from __future__ import annotations

import hashlib

import pytest

from lewm.benchmarks.go2_categorical_radial_micro_overfit import (
    LADDER_NAMESPACE,
    frame_identity,
    frame_rank,
    ladder_fit_gate,
    select_ladder_frames,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _records(count: int = 20) -> list[dict]:
    return [
        {
            "scene_id": f"scene_{index:02d}",
            "family": f"family_{index % 5}",
            "global_row": 100 + index,
            "side": "current" if index % 2 == 0 else "next",
            "image_path": f"/train/image_{index}.png",
            "image_sha256": _sha(f"image-{index}"),
            "label_shard_path": f"/train/scene_{index:02d}.npz",
            "label_shard_sha256": _sha(f"shard-{index}"),
            "label_shard_row": index,
        }
        for index in range(count)
    ]


def _presence(records: list[dict], *, all_class_index: int = 7) -> dict:
    return {
        frame_identity(record): (
            (True, True, True)
            if index == all_class_index
            else (True, True, False)
        )
        for index, record in enumerate(records)
    }


def test_ladder_selection_is_deterministic_and_scene_disjoint() -> None:
    records = _records()
    presence = _presence(records)
    first = select_ladder_frames(records, class_presence=presence)
    second = select_ladder_frames(list(reversed(records)), class_presence=presence)

    assert first["namespace"] == LADDER_NAMESPACE
    assert first["content_sha256"] == second["content_sha256"]
    assert first["prefixes"] == second["prefixes"]
    assert first["anchor"]["global_row"] == 107
    selected = first["selected_frames"]
    assert len(selected) == 16
    assert len({record["scene_id"] for record in selected}) == 16
    assert [first["prefixes"][str(size)]["frame_count"] for size in (1, 4, 16)] == [
        1,
        4,
        16,
    ]


def test_ladder_selects_lowest_rank_all_class_anchor() -> None:
    records = _records()
    presence = {
        frame_identity(record): (True, True, index in {2, 7, 13})
        for index, record in enumerate(records)
    }
    result = select_ladder_frames(records, class_presence=presence)
    candidates = [records[index] for index in (2, 7, 13)]
    expected = min(candidates, key=frame_rank)
    assert result["anchor"]["image_sha256"] == expected["image_sha256"]


def test_ladder_rejects_missing_support_duplicates_and_too_few_scenes() -> None:
    records = _records()
    presence = _presence(records)

    without_anchor = {key: (True, True, False) for key in presence}
    with pytest.raises(ValueError, match="all-class anchor"):
        select_ladder_frames(records, class_presence=without_anchor)

    incomplete = dict(presence)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="class-presence keys"):
        select_ladder_frames(records, class_presence=incomplete)

    duplicate = [dict(record) for record in records]
    duplicate[1]["image_sha256"] = duplicate[0]["image_sha256"]
    with pytest.raises(ValueError, match="image hashes"):
        select_ladder_frames(duplicate, class_presence=presence)

    short = records[:15]
    with pytest.raises(ValueError, match="16 distinct"):
        select_ladder_frames(short, class_presence=_presence(short))


def _metrics(*, nll: float, recall: float) -> dict:
    return {
        "raw_hierarchical_balanced_nll": nll,
        "class_recall": {
            "unknown": recall,
            "free": recall,
            "occupied": recall,
        },
    }


def test_one_frame_gate_is_strict() -> None:
    assert ladder_fit_gate(_metrics(nll=0.0009, recall=1.0), frame_count=1)[
        "passes"
    ]
    assert not ladder_fit_gate(_metrics(nll=0.001, recall=1.0), frame_count=1)[
        "passes"
    ]
    assert not ladder_fit_gate(_metrics(nll=0.0009, recall=0.999), frame_count=1)[
        "passes"
    ]


def test_multi_frame_gate_requires_accuracy_and_wrong_view_delta() -> None:
    metrics = _metrics(nll=0.009, recall=0.99)
    assert ladder_fit_gate(metrics, frame_count=4, wrong_view_nll=0.259)["passes"]
    assert ladder_fit_gate(metrics, frame_count=16, wrong_view_nll=0.259)[
        "passes"
    ]
    assert not ladder_fit_gate(metrics, frame_count=4, wrong_view_nll=0.2589)[
        "passes"
    ]
    assert not ladder_fit_gate(
        _metrics(nll=0.01, recall=0.99),
        frame_count=16,
        wrong_view_nll=0.5,
    )["passes"]
    with pytest.raises(ValueError, match="wrong-view"):
        ladder_fit_gate(metrics, frame_count=4)
