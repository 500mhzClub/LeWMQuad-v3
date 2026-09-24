"""Pure contracts for the categorical-radial train-only overfit ladder."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


LADDER_SCHEMA = "lewm_go2_categorical_radial_ladder_v1"
LADDER_NAMESPACE = "go2_categorical_radial_ladder_v1"
LADDER_PREFIX_SIZES = (1, 4, 16)
CLASS_NAMES = ("unknown", "free", "occupied")


def canonical_json_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def frame_identity(record: Mapping[str, Any]) -> tuple[int, str]:
    side = str(record.get("side", ""))
    if side not in {"current", "next"}:
        raise ValueError("ladder frame side must be current or next")
    return int(record["global_row"]), side


def frame_rank(record: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        [
            LADDER_NAMESPACE,
            "frame",
            str(record["scene_id"]),
            int(record["global_row"]),
            str(record["side"]),
            str(record["image_sha256"]),
        ]
    )


def scene_rank(scene_id: str) -> str:
    return canonical_json_sha256([LADDER_NAMESPACE, "scene", str(scene_id)])


def _normalized_frame(record: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "scene_id",
        "family",
        "global_row",
        "side",
        "image_path",
        "image_sha256",
        "label_shard_path",
        "label_shard_sha256",
        "label_shard_row",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"ladder frame is missing fields: {missing}")
    result = {name: record[name] for name in required}
    result["scene_id"] = str(result["scene_id"])
    result["family"] = str(result["family"])
    result["global_row"] = int(result["global_row"])
    result["side"] = str(result["side"])
    result["image_path"] = str(result["image_path"])
    result["image_sha256"] = str(result["image_sha256"])
    result["label_shard_path"] = str(result["label_shard_path"])
    result["label_shard_sha256"] = str(result["label_shard_sha256"])
    result["label_shard_row"] = int(result["label_shard_row"])
    frame_identity(result)
    for name in ("image_sha256", "label_shard_sha256"):
        value = result[name]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"ladder frame has an invalid {name}")
    return result


def select_ladder_frames(
    records: Sequence[Mapping[str, Any]],
    *,
    class_presence: Mapping[tuple[int, str], Sequence[bool]],
) -> dict[str, Any]:
    """Freeze one all-class anchor and one metadata-ranked frame per scene."""

    normalized = [_normalized_frame(record) for record in records]
    identities = [frame_identity(record) for record in normalized]
    if len(set(identities)) != len(identities):
        raise ValueError("ladder frame identities must be unique")
    image_hashes = [record["image_sha256"] for record in normalized]
    if len(set(image_hashes)) != len(image_hashes):
        raise ValueError("ladder endpoint image hashes must be unique")
    if set(class_presence) != set(identities):
        raise ValueError("class-presence keys differ from the frozen frames")

    all_class = []
    for record in normalized:
        presence = tuple(bool(value) for value in class_presence[frame_identity(record)])
        if len(presence) != len(CLASS_NAMES):
            raise ValueError("class presence must contain UNKNOWN/FREE/OCCUPIED")
        if all(presence):
            all_class.append(record)
    if not all_class:
        raise ValueError("the ladder has no all-class anchor frame")
    anchor = min(
        all_class,
        key=lambda record: (
            frame_rank(record),
            int(record["global_row"]),
            str(record["side"]),
        ),
    )

    by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in normalized:
        by_scene.setdefault(str(record["scene_id"]), []).append(record)
    anchor_scene = str(anchor["scene_id"])
    other_scene_representatives = []
    for scene_id, scene_records in by_scene.items():
        if scene_id == anchor_scene:
            continue
        representative = min(
            scene_records,
            key=lambda record: (
                frame_rank(record),
                int(record["global_row"]),
                str(record["side"]),
            ),
        )
        other_scene_representatives.append(representative)
    other_scene_representatives.sort(
        key=lambda record: (
            scene_rank(str(record["scene_id"])),
            str(record["scene_id"]),
        )
    )
    needed = max(LADDER_PREFIX_SIZES) - 1
    if len(other_scene_representatives) < needed:
        raise ValueError("the ladder requires at least 16 distinct fit scenes")
    selected = [anchor, *other_scene_representatives[:needed]]
    if len({str(record["scene_id"]) for record in selected}) != len(selected):
        raise AssertionError("ladder construction did not preserve scene disjointness")

    prefixes = {}
    for size in LADDER_PREFIX_SIZES:
        prefix = selected[:size]
        prefix_identities = [
            {
                "scene_id": str(record["scene_id"]),
                "global_row": int(record["global_row"]),
                "side": str(record["side"]),
                "image_sha256": str(record["image_sha256"]),
            }
            for record in prefix
        ]
        prefixes[str(size)] = {
            "frame_count": size,
            "frames": prefix_identities,
            "frames_sha256": canonical_json_sha256(prefix_identities),
        }
    payload = {
        "schema": LADDER_SCHEMA,
        "namespace": LADDER_NAMESPACE,
        "selection": (
            "lowest_hash_all_class_anchor_then_lowest_frame_per_"
            "metadata_ranked_distinct_scene_v1"
        ),
        "class_names": list(CLASS_NAMES),
        "anchor": prefixes["1"]["frames"][0],
        "selected_frames": selected,
        "prefixes": prefixes,
    }
    content_core = {
        name: value for name, value in payload.items() if name != "selected_frames"
    }
    payload["content_sha256"] = canonical_json_sha256(content_core)
    return payload


def ladder_fit_gate(
    metrics: Mapping[str, Any],
    *,
    frame_count: int,
    wrong_view_nll: float | None = None,
) -> dict[str, Any]:
    if frame_count not in LADDER_PREFIX_SIZES:
        raise ValueError(f"unsupported ladder frame count: {frame_count}")
    nll = float(metrics.get("raw_hierarchical_balanced_nll", float("nan")))
    recall_raw = metrics.get("class_recall")
    if not isinstance(recall_raw, Mapping):
        raise ValueError("ladder metrics lack class recall")
    recalls = {name: float(recall_raw.get(name, float("nan"))) for name in CLASS_NAMES}
    if not math.isfinite(nll) or any(not math.isfinite(value) for value in recalls.values()):
        raise ValueError("ladder metrics must be finite")
    if frame_count == 1:
        checks = {
            "balanced_nll_lt_0_001": nll < 1e-3,
            **{
                f"{name}_recall_eq_1": value == 1.0
                for name, value in recalls.items()
            },
        }
    else:
        if wrong_view_nll is None or not math.isfinite(float(wrong_view_nll)):
            raise ValueError("multi-frame ladder metrics require a wrong-view NLL")
        checks = {
            "balanced_nll_lt_0_01": nll < 0.01,
            **{
                f"{name}_recall_ge_0_99": value >= 0.99
                for name, value in recalls.items()
            },
            "wrong_view_minus_correct_nll_ge_0_25": (
                float(wrong_view_nll) - nll >= 0.25
            ),
        }
    return {
        "schema": "lewm_go2_categorical_radial_ladder_gate_v1",
        "frame_count": int(frame_count),
        "checks": checks,
        "passes": all(checks.values()),
    }


__all__ = [
    "CLASS_NAMES",
    "LADDER_NAMESPACE",
    "LADDER_PREFIX_SIZES",
    "LADDER_SCHEMA",
    "canonical_json_sha256",
    "frame_identity",
    "frame_rank",
    "ladder_fit_gate",
    "scene_rank",
    "select_ladder_frames",
]
