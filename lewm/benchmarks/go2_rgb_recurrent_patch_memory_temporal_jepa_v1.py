"""Pure identities, metrics, and gates for recurrent patch-memory JEPA V1.

This module performs no filesystem, RGB, checkpoint, model, device-discovery,
navigation, held-out, or sealed access.  It accepts already-decoded metadata
and already-computed tensors.  The future runtime is responsible for enforcing
the corresponding access and model-state boundaries before calling here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch


SCHEMA_PREFIX = "lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
PREREGISTRATION_COMMIT = "1ac341cd97ab7a7d1a1b8c46695cf2fd3382ed60"

REGISTERED_FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
ACTION_COUNT = 9
HOLD_ACTION_ID = 6

TRAIN_INDEX_ROW_COUNT = 16_000
VALIDATION_INDEX_ROW_COUNT = 2_048
TRAIN_SCHEDULE_ROW_COUNT = 4_000
TRAIN_ROWS_PER_FAMILY = 500
TRAIN_SCENE_COUNTS = dict(
    zip(
        REGISTERED_FAMILIES,
        (150, 150, 100, 250, 100, 50, 150, 50),
        strict=True,
    )
)
VALIDATION_SCENE_COUNTS = dict(
    zip(REGISTERED_FAMILIES, (22, 23, 15, 38, 15, 7, 23, 7), strict=True)
)

TRAIN_SCHEDULE_NAMESPACE = "lewm-go2-temporal-patch-gru-v1/train"
TRAIN_SCHEDULE_SHA256 = (
    "853aad503738baed6bfbba18f3ac55c4715d7f164e71d4d0ef96c861befee7fc"
)
SENTINEL_NAMESPACE = "lewm-go2-temporal-patch-gru-v1/sentinel"
SENTINEL_ROW_COUNT = 256
SENTINEL_ROWS_PER_FAMILY = 32
SENTINEL_SCENE_COUNT = 144
SENTINEL_INDICES_SHA256 = (
    "615287ba03169cfb390626d38163836d92ad1750fd5a74885e9105e56f5152ee"
)
WRONG_HISTORY_NAMESPACE = "lewm-go2-temporal-patch-gru-v1/wrong-history"
FULL_WRONG_HISTORY_DONORS_SHA256 = (
    "7bab828cc1170edc39b13e8277d3a739f97106eba4d88bed5631b27a5111823c"
)
SENTINEL_WRONG_HISTORY_DONORS_SHA256 = (
    "6d8978266e466ed191c978819d2aaa79e17773d32e4e17ac0a2542c0bb542dd4"
)
FULL_WRONG_ACTION_ROW_COUNTS = dict(
    zip(REGISTERED_FAMILIES, (248, 233, 247, 246, 248, 249, 248, 252), strict=True)
)
FULL_WRONG_ACTION_SCENE_COUNTS = dict(
    zip(REGISTERED_FAMILIES, (22, 23, 15, 38, 15, 7, 23, 7), strict=True)
)
SENTINEL_WRONG_ACTION_ROW_COUNTS = dict(
    zip(REGISTERED_FAMILIES, (32, 30, 30, 30, 31, 31, 32, 32), strict=True)
)
SENTINEL_WRONG_ACTION_SCENE_COUNTS = dict(
    zip(REGISTERED_FAMILIES, (22, 22, 15, 30, 15, 7, 23, 7), strict=True)
)

MASK_NAMESPACE = (
    "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
    "|mask|20260801"
)
GRID_SIZE = 16
TOKEN_COUNT = 256
TARGET_TOKEN_COUNT = 64
FEATURE_DIMENSION = 192

CONTROL_NAMES = (
    "persistence",
    "current_only_reset",
    "wrong_history",
    "wrong_action",
)
CAUSAL_CONTROL_NAMES = (
    "current_only_reset",
    "wrong_history",
    "wrong_action",
)
CONTROL_RATIO_MAXIMUMS = {
    "persistence": 0.95,
    "current_only_reset": 0.95,
    "wrong_history": 0.95,
    "wrong_action": 0.98,
}
CONTROL_POSITIVE_FAMILY_MINIMUMS = {
    "persistence": 6,
    "current_only_reset": 6,
    "wrong_history": 6,
    "wrong_action": 5,
}
CONTROL_BOOTSTRAP_REPLICATES = 2_000
CONTROL_BOOTSTRAP_LOWER_INDEX = 50
CONTROL_BOOTSTRAP_SEEDS = {
    "persistence": 20_260_811,
    "current_only_reset": 20_260_812,
    "wrong_history": 20_260_813,
    "wrong_action": 20_260_814,
}

OBSERVATION_UPDATES = (0, 50, 100, 200, 400)
FULL_OBSERVATION_UPDATES = (0, 200, 400)
SENTINEL_OBSERVATION_UPDATES = (0, 50, 100)
RECURRENT_EFFECTIVE_RANK_MINIMUM = 4.0
PREDICTION_HEALTH_RETENTION_MINIMUM = 0.25
PREDECESSOR_CONTROL_RATIO_MAXIMUM = 0.90
PREDECESSOR_POSITIVE_FAMILY_MINIMUM = 6
PREDECESSOR_CONTROL_NAMES = (
    "wrong_target",
    "wrong_context",
    "position_mean",
)
RAW_HEALTH_RETENTION_MINIMUM = 0.75
RAW_HEALTH_RETENTION_KEYS = (
    "online.effective_rank",
    "online.cross_sample_variance",
    "online.within_image_spatial_diversity",
    "target.effective_rank",
    "target.cross_sample_variance",
    "target.within_image_spatial_diversity",
)
PLACE_RETENTION_MINIMUM = 0.80


class TemporalJepaMetricError(RuntimeError):
    """A frozen pure identity, metric, or gate invariant failed closed."""


@dataclass(frozen=True, slots=True)
class MetadataRow:
    """The metadata-only fields used by the pure identities."""

    index: int
    role: str
    family: str
    scene_id: str
    rgb: tuple[str, ...]
    actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ControlSummary:
    """One scene-equal then family-equal temporal control comparison."""

    correct_macro_mean: float
    control_macro_mean: float
    primary_ratio: float
    advantage_macro_mean: float
    advantage_bootstrap_lower_95: float
    positive_family_count: int
    correct_by_scene: Mapping[str, float]
    control_by_scene: Mapping[str, float]
    advantage_by_scene: Mapping[str, float]
    advantage_by_family: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepresentationHealth:
    """Position-centered health for an ``N x T x 192`` token population."""

    row_count: int
    token_count: int
    feature_dimension: int
    effective_rank: float
    cross_sample_variance: float
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntegrityFacts:
    """Runtime facts that pure gate logic may consume but never manufacture."""

    access_and_accounting_exact: bool
    all_evaluated_finite: bool
    target_frozen_eval: bool
    target_gradient_tensor_count: int
    ema_count: int
    latest_training_receipt_pass: bool | None
    baseline_health_noncollapsed: bool


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    """All pure inputs for one full or sentinel observation."""

    update: int
    panel_kind: str
    panel_identity_sha256: str
    controls: Mapping[str, ControlSummary]
    recurrent_health: RepresentationHealth
    prediction_health: RepresentationHealth
    target_health: RepresentationHealth
    integrity: IntegrityFacts
    predecessor_controls: Mapping[str, ControlSummary] | None = None
    raw_health_retentions: Mapping[str, float] | None = None
    place_chance_multiple_retention: float | None = None
    target_place_rank_retention: float | None = None


@dataclass(frozen=True, slots=True)
class GateDecision:
    """One immutable continuation or terminal decision."""

    update: int
    status: str
    action: str
    passed: bool
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]
    selected_update: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_bytes(value: Any) -> bytes:
    """Return finite compact ASCII JSON bytes with sorted object keys."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise TemporalJepaMetricError("value is not canonical finite JSON") from error


def canonical_json_sha256(value: Any) -> str:
    """Hash the canonical compact JSON representation of ``value``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _rows(
    rows: Sequence[MetadataRow],
    *,
    role: str,
) -> tuple[MetadataRow, ...]:
    ordered = tuple(rows)
    if not ordered:
        raise TemporalJepaMetricError("metadata panel cannot be empty")
    for expected_index, row in enumerate(ordered):
        if (
            not isinstance(row, MetadataRow)
            or type(row.index) is not int
            or row.index != expected_index
            or row.role != role
            or row.family not in REGISTERED_FAMILIES
            or type(row.scene_id) is not str
            or not row.scene_id
            or type(row.rgb) is not tuple
            or len(row.rgb) != 7
            or any(type(value) is not str or not value for value in row.rgb)
            or type(row.actions) is not tuple
            or len(row.actions) != 6
            or any(
                type(value) is not int or not 0 <= value < ACTION_COUNT
                for value in row.actions
            )
        ):
            raise TemporalJepaMetricError("ordered metadata row identity changed")
    if {row.family for row in ordered} != set(REGISTERED_FAMILIES):
        raise TemporalJepaMetricError("metadata panel lost a registered family")
    return ordered


def _family_scene_rows(
    rows: Sequence[MetadataRow],
) -> dict[str, dict[str, tuple[MetadataRow, ...]]]:
    result: dict[str, dict[str, tuple[MetadataRow, ...]]] = {}
    for family in REGISTERED_FAMILIES:
        scenes = sorted({row.scene_id for row in rows if row.family == family})
        if not scenes:
            raise TemporalJepaMetricError(f"family {family!r} has no scenes")
        result[family] = {
            scene: tuple(
                sorted(
                    (
                        row
                        for row in rows
                        if row.family == family and row.scene_id == scene
                    ),
                    key=lambda row: row.index,
                )
            )
            for scene in scenes
        }
    return result


def _round_robin_scene_rows(
    scenes: Mapping[str, tuple[MetadataRow, ...]],
    *,
    desired_count: int,
    include_every_scene: bool,
) -> list[int]:
    names = tuple(sorted(scenes))
    if type(desired_count) is not int or desired_count <= 0:
        raise TemporalJepaMetricError("desired panel count is invalid")
    if include_every_scene and desired_count < len(names):
        raise TemporalJepaMetricError("panel cannot include every scene")
    selected: list[int] = []
    if include_every_scene:
        selected.extend(scenes[name][0].index for name in names)
        first_rank = 1
    else:
        selected.extend(scenes[name][0].index for name in names[:desired_count])
        first_rank = 1
    rank = first_rank
    while len(selected) < desired_count:
        before = len(selected)
        for name in names:
            values = scenes[name]
            if rank < len(values):
                selected.append(values[rank].index)
                if len(selected) == desired_count:
                    break
        if len(selected) == before:
            raise TemporalJepaMetricError("metadata panel lacks enough unique rows")
        rank += 1
    if len(selected) != desired_count or len(set(selected)) != desired_count:
        raise TemporalJepaMetricError("round-robin panel identity is invalid")
    return selected


def _hash_order(namespace: str, indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        sorted(
            indices,
            key=lambda index: (
                hashlib.sha256(f"{namespace}/{index}".encode("ascii")).digest(),
                index,
            ),
        )
    )


def build_training_schedule(
    rows: Sequence[MetadataRow],
    *,
    rows_per_family: int = TRAIN_ROWS_PER_FAMILY,
) -> tuple[int, ...]:
    """Build the preregistered family-equal, all-scene train schedule."""

    ordered = _rows(rows, role="train")
    grouped = _family_scene_rows(ordered)
    selected: list[int] = []
    for family in REGISTERED_FAMILIES:
        selected.extend(
            _round_robin_scene_rows(
                grouped[family],
                desired_count=rows_per_family,
                include_every_scene=True,
            )
        )
    result = _hash_order(TRAIN_SCHEDULE_NAMESPACE, selected)
    if (
        len(result) != rows_per_family * len(REGISTERED_FAMILIES)
        or len(set(result)) != len(result)
    ):
        raise TemporalJepaMetricError("training schedule cardinality changed")
    return result


def build_sentinel_indices(
    rows: Sequence[MetadataRow],
    *,
    rows_per_family: int = SENTINEL_ROWS_PER_FAMILY,
) -> tuple[int, ...]:
    """Build the preregistered family-balanced validation sentinel."""

    ordered = _rows(rows, role="val")
    grouped = _family_scene_rows(ordered)
    selected: list[int] = []
    for family in REGISTERED_FAMILIES:
        scenes = grouped[family]
        if len(scenes) >= rows_per_family:
            family_selected = [
                scenes[name][0].index
                for name in tuple(sorted(scenes))[:rows_per_family]
            ]
        else:
            family_selected = _round_robin_scene_rows(
                scenes,
                desired_count=rows_per_family,
                include_every_scene=True,
            )
        selected.extend(family_selected)
    result = _hash_order(SENTINEL_NAMESPACE, selected)
    if (
        len(result) != rows_per_family * len(REGISTERED_FAMILIES)
        or len(set(result)) != len(result)
    ):
        raise TemporalJepaMetricError("sentinel cardinality changed")
    return result


def build_wrong_history_donor_indices(
    rows: Sequence[MetadataRow],
    *,
    selected_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Build frozen same-family, different-scene wrong-history donors."""

    ordered = _rows(rows, role="val")
    if selected_indices is None:
        selected = tuple(range(len(ordered)))
    else:
        selected = tuple(selected_indices)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(
                type(index) is not int or not 0 <= index < len(ordered)
                for index in selected
            )
        ):
            raise TemporalJepaMetricError("wrong-history selected indices changed")
    family_indices = {
        family: tuple(row.index for row in ordered if row.family == family)
        for family in REGISTERED_FAMILIES
    }
    donors: list[int] = []
    for row_index in selected:
        row = ordered[row_index]
        eligible = tuple(
            donor
            for donor in family_indices[row.family]
            if ordered[donor].scene_id != row.scene_id
        )
        if not eligible:
            raise TemporalJepaMetricError("wrong-history donor is unavailable")
        donors.append(
            min(
                eligible,
                key=lambda donor: (
                    hashlib.sha256(
                        (
                            f"{WRONG_HISTORY_NAMESPACE}/"
                            f"{row.index}/{donor}"
                        ).encode("ascii")
                    ).digest(),
                    donor,
                ),
            )
        )
    if any(
        ordered[donor].family != ordered[row].family
        or ordered[donor].scene_id == ordered[row].scene_id
        for row, donor in zip(selected, donors, strict=True)
    ):
        raise TemporalJepaMetricError("wrong-history donor constraints changed")
    return tuple(donors)


def wrong_action_eligible_indices(
    rows: Sequence[MetadataRow],
    *,
    selected_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return factual rows whose outgoing action is not HOLD."""

    ordered = _rows(rows, role="val")
    selected = (
        tuple(range(len(ordered)))
        if selected_indices is None
        else tuple(selected_indices)
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(
            type(index) is not int or not 0 <= index < len(ordered)
            for index in selected
        )
    ):
        raise TemporalJepaMetricError("wrong-action selected indices changed")
    return tuple(
        index for index in selected if ordered[index].actions[2] != HOLD_ACTION_ID
    )


def family_row_scene_counts(
    rows: Sequence[MetadataRow],
    indices: Sequence[int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Count selected rows and distinct scenes in registered family order."""

    ordered = tuple(rows)
    selected = tuple(indices)
    if len(set(selected)) != len(selected) or any(
        type(index) is not int or not 0 <= index < len(ordered) for index in selected
    ):
        raise TemporalJepaMetricError("selected count indices changed")
    row_counts = {
        family: sum(ordered[index].family == family for index in selected)
        for family in REGISTERED_FAMILIES
    }
    scene_counts = {
        family: len(
            {
                ordered[index].scene_id
                for index in selected
                if ordered[index].family == family
            }
        )
        for family in REGISTERED_FAMILIES
    }
    return row_counts, scene_counts


def mask_indices(role: str, row_index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return exact predecessor target and visible token indices."""

    counts = {"train": TRAIN_INDEX_ROW_COUNT, "val": VALIDATION_INDEX_ROW_COUNT}
    if role not in counts or type(row_index) is not int:
        raise TemporalJepaMetricError("mask role or row-index type changed")
    if not 0 <= row_index < counts[role]:
        raise TemporalJepaMetricError("mask row index left its frozen role")
    target: set[int] = set()
    for quadrant, (base_row, base_column) in enumerate(
        ((0, 0), (0, 8), (8, 0), (8, 8))
    ):
        digest = hashlib.sha256(
            f"{MASK_NAMESPACE}|{role}|{row_index}|{quadrant}".encode("ascii")
        ).digest()
        row_offset = int.from_bytes(digest[0:4], "big") % 5
        column_offset = int.from_bytes(digest[4:8], "big") % 5
        target.update(
            (base_row + row_offset + local_row) * GRID_SIZE
            + base_column
            + column_offset
            + local_column
            for local_row in range(4)
            for local_column in range(4)
        )
    targets = tuple(sorted(target))
    visible = tuple(index for index in range(TOKEN_COUNT) if index not in target)
    if len(targets) != TARGET_TOKEN_COUNT or len(visible) != 192:
        raise TemporalJepaMetricError("mask cardinality changed")
    return targets, visible


def batched_mask_indices(
    role: str,
    row_indices: Sequence[int],
    *,
    device: Any = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize exact masks as target-Bx64 and visible-Bx192 tensors."""

    ordered = tuple(row_indices)
    if not ordered:
        raise TemporalJepaMetricError("mask batch cannot be empty")
    masks = tuple(mask_indices(role, index) for index in ordered)
    return (
        torch.tensor(
            [value[0] for value in masks],
            dtype=torch.long,
            device=torch.device(device),
        ),
        torch.tensor(
            [value[1] for value in masks],
            dtype=torch.long,
            device=torch.device(device),
        ),
    )


def validation_panel_identity(
    rows: Sequence[MetadataRow],
    row_indices: Sequence[int],
    donor_indices: Sequence[int],
    wrong_action_indices: Sequence[int],
) -> str:
    """Bind rows, masks, donors, and wrong-action eligibility as one panel."""

    ordered = _rows(rows, role="val")
    selected = tuple(row_indices)
    donors = tuple(donor_indices)
    eligible = tuple(wrong_action_indices)
    if len(selected) != len(donors):
        raise TemporalJepaMetricError("panel donor cardinality changed")
    if not set(eligible).issubset(selected):
        raise TemporalJepaMetricError("wrong-action panel escaped its parent panel")
    masks = [mask_indices("val", index)[0] for index in selected]
    return canonical_json_sha256(
        {
            "row_indices": selected,
            "target_indices_sha256": canonical_json_sha256(masks),
            "wrong_history_donor_indices": donors,
            "wrong_action_eligible_indices": eligible,
        }
    )


def expected_training_accounting(update: int) -> dict[str, int]:
    """Return exact training-only accounting through a completed update."""

    if type(update) is not int or not 0 <= update <= 400:
        raise TemporalJepaMetricError("accounting update left the registered cap")
    return {
        "updates": update,
        "sequence_rows": 10 * update,
        "rgb_frame_presentations": 40 * update,
        "online_encoder_frame_calls": 30 * update,
        "ema_target_encoder_frame_calls": 10 * update,
        "microbatch_graphs": 5 * update,
        "backward_calls": 5 * update,
        "global_gradient_clips": update,
        "optimizer_steps": update,
        "ema_steps": update,
    }


def normalized_half_squared_energy(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return one normalized half-squared 64-token energy per row."""

    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or prediction.shape != target.shape
        or prediction.ndim != 3
        or tuple(prediction.shape[1:]) != (TARGET_TOKEN_COUNT, FEATURE_DIMENSION)
        or not prediction.is_floating_point()
        or not target.is_floating_point()
        or not bool(torch.isfinite(prediction).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise TemporalJepaMetricError("temporal energy operands are invalid")
    prediction = torch.nn.functional.normalize(prediction, dim=-1, eps=1.0e-8)
    target = torch.nn.functional.normalize(target, dim=-1, eps=1.0e-8)
    energy = 0.5 * (prediction - target).square().sum(dim=-1).mean(dim=-1)
    if not bool(torch.isfinite(energy).all()) or bool((energy < 0.0).any()):
        raise TemporalJepaMetricError("temporal energy is invalid")
    return energy.detach().to(device="cpu", dtype=torch.float64)


def _energy_vector(
    value: torch.Tensor,
    *,
    name: str,
    expected_count: int | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TemporalJepaMetricError(f"{name} must be a floating tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if (
        result.numel() < 1
        or (expected_count is not None and result.numel() != expected_count)
        or not bool(torch.isfinite(result).all())
        or bool((result < 0.0).any())
    ):
        raise TemporalJepaMetricError(f"{name} energy vector is invalid")
    return result


def summarize_control(
    real_energy: torch.Tensor,
    control_energy: torch.Tensor,
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    control_name: str,
) -> ControlSummary:
    """Summarize one frozen temporal control and seeded scene bootstrap."""

    if control_name not in CONTROL_NAMES:
        raise TemporalJepaMetricError("unregistered temporal control")
    real = _energy_vector(real_energy, name="real")
    control = _energy_vector(
        control_energy, name=control_name, expected_count=real.numel()
    )
    scenes = tuple(scene_ids)
    families = tuple(family_ids)
    if (
        len(scenes) != real.numel()
        or len(families) != real.numel()
        or any(type(value) is not str or not value for value in scenes + families)
        or set(families) != set(REGISTERED_FAMILIES)
    ):
        raise TemporalJepaMetricError("control metadata population changed")
    scene_family: dict[str, str] = {}
    for scene, family in zip(scenes, families, strict=True):
        if family not in REGISTERED_FAMILIES:
            raise TemporalJepaMetricError("control family is unregistered")
        previous = scene_family.setdefault(scene, family)
        if previous != family:
            raise TemporalJepaMetricError("one scene belongs to multiple families")

    correct_by_scene: dict[str, float] = {}
    control_by_scene: dict[str, float] = {}
    advantage_by_scene: dict[str, float] = {}
    for scene in sorted(scene_family):
        indices = [index for index, value in enumerate(scenes) if value == scene]
        correct_mean = float(real[indices].mean())
        control_mean = float(control[indices].mean())
        correct_by_scene[scene] = correct_mean
        control_by_scene[scene] = control_mean
        advantage_by_scene[scene] = control_mean - correct_mean

    correct_by_family: dict[str, float] = {}
    control_by_family: dict[str, float] = {}
    advantage_by_family: dict[str, float] = {}
    family_scenes: dict[str, tuple[str, ...]] = {}
    for family in REGISTERED_FAMILIES:
        names = tuple(
            scene for scene in sorted(scene_family) if scene_family[scene] == family
        )
        if not names:
            raise TemporalJepaMetricError("control family has no scenes")
        family_scenes[family] = names
        correct_by_family[family] = sum(correct_by_scene[name] for name in names) / len(
            names
        )
        control_by_family[family] = sum(control_by_scene[name] for name in names) / len(
            names
        )
        advantage_by_family[family] = sum(
            advantage_by_scene[name] for name in names
        ) / len(names)

    correct_macro = sum(correct_by_family.values()) / len(REGISTERED_FAMILIES)
    control_macro = sum(control_by_family.values()) / len(REGISTERED_FAMILIES)
    if not control_macro > 0.0:
        raise TemporalJepaMetricError("control macro mean is nonpositive")
    advantage_macro = sum(advantage_by_family.values()) / len(REGISTERED_FAMILIES)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(CONTROL_BOOTSTRAP_SEEDS[control_name])
    bootstrap = torch.zeros(CONTROL_BOOTSTRAP_REPLICATES, dtype=torch.float64)
    for family in REGISTERED_FAMILIES:
        values = torch.tensor(
            [advantage_by_scene[scene] for scene in family_scenes[family]],
            dtype=torch.float64,
        )
        count = values.numel()
        sampled = torch.randint(
            count,
            (CONTROL_BOOTSTRAP_REPLICATES, count),
            generator=generator,
        )
        bootstrap += values[sampled].mean(dim=1)
    bootstrap /= len(REGISTERED_FAMILIES)
    lower = float(bootstrap.sort().values[CONTROL_BOOTSTRAP_LOWER_INDEX])

    return ControlSummary(
        correct_macro_mean=correct_macro,
        control_macro_mean=control_macro,
        primary_ratio=correct_macro / control_macro,
        advantage_macro_mean=advantage_macro,
        advantage_bootstrap_lower_95=lower,
        positive_family_count=sum(
            value > 0.0 for value in advantage_by_family.values()
        ),
        correct_by_scene=correct_by_scene,
        control_by_scene=control_by_scene,
        advantage_by_scene=advantage_by_scene,
        advantage_by_family=advantage_by_family,
    )


def representation_health(
    tokens: torch.Tensor,
    *,
    expected_token_count: int | None = None,
) -> RepresentationHealth:
    """Compute the preregistered position-centered covariance health."""

    if (
        not isinstance(tokens, torch.Tensor)
        or not tokens.is_floating_point()
        or tokens.ndim != 3
        or tokens.shape[0] < 1
        or tokens.shape[1] < 1
        or tokens.shape[2] != FEATURE_DIMENSION
        or (
            expected_token_count is not None
            and tokens.shape[1] != expected_token_count
        )
        or not bool(torch.isfinite(tokens).all())
    ):
        raise TemporalJepaMetricError("representation-health tensor is invalid")
    value = tokens.detach().to(device="cpu", dtype=torch.float64)
    row_count, token_count, feature_dimension = map(int, value.shape)
    denominator = row_count * token_count - 1
    if denominator <= 0:
        raise TemporalJepaMetricError("representation-health population is too small")
    centered = value - value.mean(dim=0, keepdim=True)
    flattened = centered.reshape(-1, feature_dimension)
    covariance = flattened.T.mm(flattened) / denominator
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / total
        entropy = -(
            probabilities
            * probabilities.clamp_min(1.0e-12).log()
        ).sum()
        effective_rank = float(entropy.exp())
    variance = float(
        centered.square().sum() / (row_count * token_count * feature_dimension)
    )
    finite = math.isfinite(effective_rank) and math.isfinite(variance)
    if not finite or effective_rank < 0.0 or variance < 0.0:
        raise TemporalJepaMetricError("representation health is nonfinite")
    return RepresentationHealth(
        row_count=row_count,
        token_count=token_count,
        feature_dimension=feature_dimension,
        effective_rank=effective_rank,
        cross_sample_variance=variance,
        finite=finite,
    )


def recurrent_temporal_change(
    penultimate_state: torch.Tensor,
    final_state: torch.Tensor,
) -> float:
    """Mean squared change between GRU context steps two and three."""

    if (
        not isinstance(penultimate_state, torch.Tensor)
        or not isinstance(final_state, torch.Tensor)
        or penultimate_state.shape != final_state.shape
        or penultimate_state.ndim != 3
        or tuple(penultimate_state.shape[1:]) != (TOKEN_COUNT, FEATURE_DIMENSION)
        or not penultimate_state.is_floating_point()
        or not final_state.is_floating_point()
        or not bool(torch.isfinite(penultimate_state).all())
        or not bool(torch.isfinite(final_state).all())
    ):
        raise TemporalJepaMetricError("recurrent temporal-change operands are invalid")
    value = float(
        (
            final_state.detach().to(device="cpu", dtype=torch.float64)
            - penultimate_state.detach().to(device="cpu", dtype=torch.float64)
        )
        .square()
        .mean()
    )
    if not math.isfinite(value) or value < 0.0:
        raise TemporalJepaMetricError("recurrent temporal change is invalid")
    return value


def _control_thresholds(
    controls: Mapping[str, ControlSummary],
) -> dict[str, bool]:
    if set(controls) != set(CONTROL_NAMES) or any(
        not isinstance(controls[name], ControlSummary) for name in CONTROL_NAMES
    ):
        raise TemporalJepaMetricError("temporal control inventory changed")
    checks: dict[str, bool] = {}
    for name in CONTROL_NAMES:
        value = controls[name]
        checks[f"{name}_ratio"] = (
            math.isfinite(value.primary_ratio)
            and value.primary_ratio <= CONTROL_RATIO_MAXIMUMS[name]
        )
        checks[f"{name}_bootstrap"] = (
            math.isfinite(value.advantage_bootstrap_lower_95)
            and value.advantage_bootstrap_lower_95 > 0.0
        )
        checks[f"{name}_family_breadth"] = (
            value.positive_family_count
            >= CONTROL_POSITIVE_FAMILY_MINIMUMS[name]
        )
    return checks


def _prediction_health_checks(
    observation: TemporalObservation,
) -> dict[str, bool]:
    prediction = observation.prediction_health
    target = observation.target_health
    recurrent = observation.recurrent_health
    finite = all(
        value.finite
        and math.isfinite(value.effective_rank)
        and math.isfinite(value.cross_sample_variance)
        for value in (prediction, target, recurrent)
    )
    target_positive = (
        target.effective_rank > 0.0 and target.cross_sample_variance > 0.0
    )
    return {
        "temporal_health_finite": finite,
        "recurrent_effective_rank": (
            recurrent.effective_rank >= RECURRENT_EFFECTIVE_RANK_MINIMUM
        ),
        "prediction_effective_rank_retention": (
            target_positive
            and prediction.effective_rank / target.effective_rank
            >= PREDICTION_HEALTH_RETENTION_MINIMUM
        ),
        "prediction_variance_retention": (
            target_positive
            and prediction.cross_sample_variance / target.cross_sample_variance
            >= PREDICTION_HEALTH_RETENTION_MINIMUM
        ),
    }


def _predecessor_checks(observation: TemporalObservation) -> dict[str, bool]:
    controls = observation.predecessor_controls
    if controls is None or set(controls) != set(PREDECESSOR_CONTROL_NAMES):
        return {"predecessor_spatial_controls": False}
    checks = {
        f"predecessor_{name}": (
            isinstance(value, ControlSummary)
            and value.primary_ratio <= PREDECESSOR_CONTROL_RATIO_MAXIMUM
            and value.advantage_bootstrap_lower_95 > 0.0
            and value.positive_family_count
            >= PREDECESSOR_POSITIVE_FAMILY_MINIMUM
        )
        for name, value in controls.items()
    }
    checks["predecessor_spatial_controls"] = all(checks.values())
    return checks


def temporal_thresholds_pass(observation: TemporalObservation) -> bool:
    """Return the sentinel-computable subset of the qualification gate."""

    checks = {
        **_control_thresholds(observation.controls),
        **_prediction_health_checks(observation),
    }
    return all(checks.values())


def observation_survival_checks(
    observation: TemporalObservation,
) -> dict[str, bool]:
    """Evaluate the mandatory observation-survival predicate."""

    if (
        observation.update not in OBSERVATION_UPDATES
        or observation.panel_kind not in {"full", "sentinel"}
        or type(observation.panel_identity_sha256) is not str
        or len(observation.panel_identity_sha256) != 64
    ):
        raise TemporalJepaMetricError("observation identity changed")
    integrity = observation.integrity
    panel_schedule_exact = (
        (observation.update == 0 and observation.panel_kind in {"full", "sentinel"})
        or (
            observation.update in {50, 100}
            and observation.panel_kind == "sentinel"
        )
        or (
            observation.update in {200, 400}
            and observation.panel_kind == "full"
        )
    )
    checks = {
        "registered_panel_schedule": panel_schedule_exact,
        "access_and_accounting_exact": integrity.access_and_accounting_exact,
        "all_evaluated_finite": integrity.all_evaluated_finite,
        "target_frozen_eval": integrity.target_frozen_eval,
        "target_gradient_tensor_count_zero": (
            integrity.target_gradient_tensor_count == 0
        ),
        "ema_count_exact": integrity.ema_count == observation.update,
        **_prediction_health_checks(observation),
    }
    if observation.update == 0:
        checks["no_update_zero_training_receipt"] = (
            integrity.latest_training_receipt_pass is None
        )
        checks["baseline_health_noncollapsed"] = (
            integrity.baseline_health_noncollapsed
        )
        checks.update(_predecessor_checks(observation))
    else:
        checks["latest_training_receipt_pass"] = (
            integrity.latest_training_receipt_pass is True
        )
    if observation.panel_kind == "full" and observation.update > 0:
        checks.update(_predecessor_checks(observation))
        retentions = observation.raw_health_retentions
        checks["raw_health_retention"] = bool(
            retentions is not None
            and set(retentions) == set(RAW_HEALTH_RETENTION_KEYS)
            and all(
                type(value) in {int, float}
                and math.isfinite(float(value))
                and float(value) >= RAW_HEALTH_RETENTION_MINIMUM
                for value in retentions.values()
            )
        )
        checks["place_chance_multiple_retention"] = bool(
            observation.place_chance_multiple_retention is not None
            and math.isfinite(observation.place_chance_multiple_retention)
            and observation.place_chance_multiple_retention
            >= PLACE_RETENTION_MINIMUM
        )
        checks["target_place_rank_retention"] = bool(
            observation.target_place_rank_retention is not None
            and math.isfinite(observation.target_place_rank_retention)
            and observation.target_place_rank_retention
            >= PLACE_RETENTION_MINIMUM
        )
    return checks


def observation_survives(observation: TemporalObservation) -> bool:
    """Return whether the mandatory observation-survival predicate passes."""

    return all(observation_survival_checks(observation).values())


def qualification_checks(observation: TemporalObservation) -> dict[str, bool]:
    """Evaluate the complete full-validation qualification gate."""

    if observation.panel_kind != "full" or observation.update not in (200, 400):
        raise TemporalJepaMetricError(
            "qualification requires update 200 or 400 full panel"
        )
    return {
        **observation_survival_checks(observation),
        **_control_thresholds(observation.controls),
    }


def qualifies(observation: TemporalObservation) -> bool:
    """Return whether one registered full observation qualifies."""

    return all(qualification_checks(observation).values())


def observation_gate(
    observation: TemporalObservation,
) -> dict[str, Any]:
    """Return one JSONable observation survival/qualification record."""

    survival = observation_survival_checks(observation)
    qualification: dict[str, bool] | None = None
    if observation.panel_kind == "full" and observation.update in (200, 400):
        qualification = qualification_checks(observation)
    return {
        "schema": f"{SCHEMA_PREFIX}_observation_gate_v1",
        "update": observation.update,
        "panel_kind": observation.panel_kind,
        "panel_identity_sha256": observation.panel_identity_sha256,
        "survival_checks": survival,
        "observation_survives": all(survival.values()),
        "temporal_thresholds_pass": temporal_thresholds_pass(observation),
        "qualification_checks": qualification,
        "perception_temporal_qualified": (
            qualification is not None and all(qualification.values())
        ),
    }


def _compatible(left: TemporalObservation, right: TemporalObservation) -> bool:
    return (
        left.panel_kind == right.panel_kind
        and left.panel_identity_sha256 == right.panel_identity_sha256
    )


def _worst_causal_ratio(observation: TemporalObservation) -> float:
    return max(
        observation.controls[name].primary_ratio for name in CAUSAL_CONTROL_NAMES
    )


def _minimum_causal_family_breadth(observation: TemporalObservation) -> int:
    return min(
        observation.controls[name].positive_family_count
        for name in CAUSAL_CONTROL_NAMES
    )


def _decision(
    observation: TemporalObservation,
    *,
    status: str,
    action: str,
    passed: bool,
    checks: Mapping[str, bool],
    selected_update: int | None = None,
) -> GateDecision:
    return GateDecision(
        update=observation.update,
        status=status,
        action=action,
        passed=passed,
        checks=dict(checks),
        failed_checks=tuple(
            sorted(name for name, value in checks.items() if not value)
        ),
        selected_update=selected_update,
    )


def continuation_decision(
    current: TemporalObservation,
    *,
    update_zero: TemporalObservation,
    previous: TemporalObservation | None = None,
) -> GateDecision:
    """Apply exact compatible-panel continuation and terminal rules."""

    survival = observation_survival_checks(current)
    if not all(survival.values()):
        return _decision(
            current,
            status="FAIL_OBSERVATION_SURVIVAL",
            action="TERMINAL",
            passed=False,
            checks=survival,
        )
    if current.update == 0:
        return _decision(
            current,
            status="CONTINUE_FROM_BASELINE",
            action="CONTINUE",
            passed=True,
            checks=survival,
        )
    if current.update == 50:
        if update_zero.update != 0 or not _compatible(current, update_zero):
            raise TemporalJepaMetricError("update-50 comparison panel changed")
        ratio_improved = any(
            update_zero.controls[name].primary_ratio
            - current.controls[name].primary_ratio
            >= 0.01
            for name in CAUSAL_CONTROL_NAMES
        )
        breadth_increased = (
            _minimum_causal_family_breadth(current)
            > _minimum_causal_family_breadth(update_zero)
        )
        checks = {
            **survival,
            "ratio_or_breadth_progress": ratio_improved or breadth_increased,
        }
        passed = all(checks.values())
        return _decision(
            current,
            status="CONTINUE_UPDATE_50" if passed else "FAIL_UPDATE_50_CONTINUATION",
            action="CONTINUE" if passed else "TERMINAL",
            passed=passed,
            checks=checks,
        )
    if current.update == 100:
        if (
            previous is None
            or previous.update != 50
            or not _compatible(current, previous)
        ):
            raise TemporalJepaMetricError("update-100 comparison panel changed")
        worst_improved = (
            _worst_causal_ratio(previous) - _worst_causal_ratio(current) >= 0.02
        )
        breadth_increased = (
            _minimum_causal_family_breadth(current)
            > _minimum_causal_family_breadth(previous)
        )
        checks = {
            **survival,
            "trend_or_temporal_thresholds": (
                worst_improved
                or breadth_increased
                or temporal_thresholds_pass(current)
            ),
        }
        passed = all(checks.values())
        return _decision(
            current,
            status="CONTINUE_UPDATE_100" if passed else "FAIL_UPDATE_100_CONTINUATION",
            action="CONTINUE" if passed else "TERMINAL",
            passed=passed,
            checks=checks,
        )
    if current.update == 200:
        if qualifies(current):
            return _decision(
                current,
                status="PASS_TEMPORAL_QUALIFIED",
                action="SELECT_AND_STOP",
                passed=True,
                checks=qualification_checks(current),
                selected_update=200,
            )
        if update_zero.update != 0 or not _compatible(current, update_zero):
            raise TemporalJepaMetricError("update-200 comparison panel changed")
        all_below_one = all(
            current.controls[name].primary_ratio < 1.0
            for name in CAUSAL_CONTROL_NAMES
        )
        trend = (
            _worst_causal_ratio(update_zero) - _worst_causal_ratio(current)
            >= 0.02
            and _minimum_causal_family_breadth(current)
            > _minimum_causal_family_breadth(update_zero)
        )
        checks = {
            **survival,
            "causal_separation_or_progress": all_below_one or trend,
        }
        passed = all(checks.values())
        return _decision(
            current,
            status="CONTINUE_UPDATE_200" if passed else "FAIL_UPDATE_200_CONTINUATION",
            action="CONTINUE" if passed else "TERMINAL",
            passed=passed,
            checks=checks,
        )
    if current.update == 400:
        checks = qualification_checks(current)
        passed = all(checks.values())
        return _decision(
            current,
            status=(
                "PASS_TEMPORAL_QUALIFIED"
                if passed
                else "FAIL_SCIENTIFIC_NO_QUALIFYING_CHECKPOINT"
            ),
            action="SELECT_AND_STOP" if passed else "TERMINAL",
            passed=passed,
            checks=checks,
            selected_update=400 if passed else None,
        )
    raise TemporalJepaMetricError("unregistered continuation update")


def continuation_gate(
    current: TemporalObservation,
    *,
    update_zero: TemporalObservation,
    previous: TemporalObservation | None = None,
) -> dict[str, Any]:
    """Return the exact continuation decision as a JSONable mapping."""

    return continuation_decision(
        current,
        update_zero=update_zero,
        previous=previous,
    ).to_dict()


def select_qualified_observation(
    observations: Sequence[TemporalObservation],
) -> TemporalObservation | None:
    """Select update 200 if qualified, otherwise the qualifying update 400."""

    indexed = {value.update: value for value in observations}
    if len(indexed) != len(tuple(observations)):
        raise TemporalJepaMetricError("duplicate observation update")
    for update in (200, 400):
        value = indexed.get(update)
        if value is not None and qualifies(value):
            return value
    return None
