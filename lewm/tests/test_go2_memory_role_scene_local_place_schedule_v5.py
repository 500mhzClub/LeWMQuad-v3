from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from lewm.datasets.go2_memory_role_place_triplets_v1 import (
    RGBReference,
    SCHEMA,
    PlaceTripletRow,
    canonical_json_sha256,
    decode_index_row,
)
from lewm.datasets.go2_memory_role_scene_local_place_schedule_v5 import (
    EXPECTED_SOURCE_ROWS_BY_FAMILY_V5,
    FAMILIES_V5,
    SceneLocalPlaceScheduleError,
    build_scene_local_place_schedule_v5,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _reference(index: int, role: str) -> RGBReference:
    return RGBReference(
        endpoint_identity_sha256=_sha(f"endpoint-{index}-{role}"),
        rgb_path=(
            ".generated/go2_render_selected_v04/scenes/"
            f"scene_{index:016x}/rgb/frame_{index:06d}_env_00.png"
        ),
        image_sha256=_sha(f"image-{index}-{role}"),
    )


def _synthetic_frozen_rows() -> tuple[PlaceTripletRow, ...]:
    rows: list[PlaceTripletRow] = []
    for family in FAMILIES_V5:
        for family_index in range(EXPECTED_SOURCE_ROWS_BY_FAMILY_V5[family]):
            index = len(rows)
            rows.append(
                PlaceTripletRow(
                    index=index,
                    role="train",
                    family=family,
                    scene_id=f"scene_{family}",
                    anchor=_reference(index, "anchor"),
                    positive=_reference(index, "positive"),
                    negative=_reference(index, "negative"),
                    content_sha256=_sha(f"content-{index}"),
                    place_identity_sha256=_sha(
                        f"place-{family}-{family_index % 8}"
                    ),
                )
            )
    return tuple(rows)


def test_exact_schedule_is_deterministic_balanced_and_metadata_only() -> None:
    rows = _synthetic_frozen_rows()

    groups, receipt = build_scene_local_place_schedule_v5(rows)
    repeated_groups, repeated_receipt = build_scene_local_place_schedule_v5(rows)

    assert groups == repeated_groups
    assert receipt == repeated_receipt
    assert len(groups) == 800
    assert sum(len(group.rows) for group in groups) == 3_200
    assert all(len({row.scene_id for row in group.rows}) == 1 for group in groups)
    assert all(
        len({row.place_identity_sha256 for row in group.rows}) == 4
        for group in groups
    )
    assert {
        family: sum(group.family == family for group in groups)
        for family in FAMILIES_V5
    } == {family: 100 for family in FAMILIES_V5}
    assert receipt["accounting"]["repeated_row_presentation_count"] == 323
    assert receipt["accounting"]["dropped_source_row_count"] == 323
    assert receipt["access_ledger"] == {
        "rgb_byte_open_count": 0,
        "rgb_decode_count": 0,
        "privileged_cell_yaw_emitted_count": 0,
        "heldout_or_sealed_open_count": 0,
    }


def test_schedule_rejects_a_family_without_four_places_in_one_scene() -> None:
    rows = list(_synthetic_frozen_rows())
    family = FAMILIES_V5[0]
    one_place = _sha("only-place")
    rows = [
        replace(row, place_identity_sha256=one_place)
        if row.family == family
        else row
        for row in rows
    ]

    with pytest.raises(
        SceneLocalPlaceScheduleError,
        match="no scene with four distinct place identities",
    ):
        build_scene_local_place_schedule_v5(rows)


def _encoded_row(*, cell_id: int) -> dict[str, object]:
    scene_id = "scene_train"

    def reference(index: int) -> dict[str, str]:
        return {
            "endpoint_identity_sha256": _sha(f"raw-endpoint-{index}"),
            "rgb_path": (
                ".generated/go2_render_selected_v04/scenes/"
                "scene_0123456789abcdef/rgb/"
                f"frame_{index:06d}_env_00.png"
            ),
            "image_sha256": _sha(f"raw-image-{index}"),
        }

    core: dict[str, object] = {
        "schema": SCHEMA,
        "role": "train",
        "family": FAMILIES_V5[0],
        "scene_id": scene_id,
        "anchor": reference(0),
        "positive": reference(1),
        "negative": reference(2),
        "selection_proof": {
            "anchor": {
                "cell_id": cell_id,
                "yaw_bin": 2,
                "env_index": 0,
                "episode_id": "episode_a",
                "timestamp_ns": 0,
            },
            "positive": {
                "cell_id": cell_id,
                "yaw_bin": 2,
                "env_index": 1,
                "episode_id": "episode_b",
                "timestamp_ns": 0,
            },
            "negative": {
                "cell_id": cell_id + 1,
                "yaw_bin": 2,
                "env_index": 0,
                "episode_id": "episode_c",
                "timestamp_ns": 0,
            },
            "positive_separation": "different_stream",
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def test_adapter_emits_only_an_opaque_place_identity() -> None:
    decoded = decode_index_row(_encoded_row(cell_id=7), index=0, role="train")
    changed = decode_index_row(_encoded_row(cell_id=8), index=0, role="train")

    assert decoded.place_identity_sha256 == canonical_json_sha256(
        {
            "schema": "lewm_go2_memory_role_place_identity_v1",
            "family": FAMILIES_V5[0],
            "scene_id": "scene_train",
            "cell_id": 7,
            "yaw_bin": 2,
        }
    )
    assert decoded.place_identity_sha256 != changed.place_identity_sha256
    assert not hasattr(decoded, "cell_id")
    assert not hasattr(decoded, "yaw_bin")
