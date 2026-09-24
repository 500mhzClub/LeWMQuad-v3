"""Pure masks and metrics for the single-frame masked spatial JEPA V1.

This module has no filesystem, RGB, checkpoint, or model access.  It accepts
already-computed tensors and non-privileged row metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch


SCHEMA_PREFIX = "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
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
TRAIN_ROW_COUNT = 16_000
VALIDATION_ROW_COUNT = 2_048
OBSERVATION_UPDATES = (0, 250, 500, 750, 1_000)

MASK_NAMESPACE = (
    "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
    "|mask|20260801"
)
GRID_SIZE = 16
TOKEN_COUNT = GRID_SIZE * GRID_SIZE
QUADRANT_SIZE = 8
BLOCK_SIZE = 4
TARGET_TOKEN_COUNT = 64
VISIBLE_TOKEN_COUNT = 192
FEATURE_DIMENSION = 192

CONTROL_BOOTSTRAP_REPLICATES = 2_000
CONTROL_BOOTSTRAP_LOWER_INDEX = 50
CONTROL_BOOTSTRAP_SEEDS = {
    "wrong_target": 20_260_802,
    "wrong_context": 20_260_803,
    "position_mean": 20_260_804,
}
PLACE_BOOTSTRAP_SEED = 20_260_805
PLACE_SELECTION_ROW_COUNT = 320
PLACE_FAMILY_ROW_COUNTS = dict(
    zip(REGISTERED_FAMILIES, (32, 48, 32, 32, 64, 64, 20, 28), strict=True)
)
PLACE_MINIMUM_CANDIDATES = 32
PLACE_MAXIMUM_CANDIDATES = 64


class MaskedSpatialMetricError(RuntimeError):
    """A frozen mask or pure metric invariant failed closed."""


@dataclass(frozen=True, slots=True)
class ControlSummary:
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
class RawRepresentationHealth:
    image_count: int
    token_count: int
    feature_dimension: int
    effective_rank: float
    cross_sample_variance: float
    within_image_spatial_diversity: float
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mask_indices(role: str, row_index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the exact sorted target and visible indices for one H6 row."""

    row_counts = {"train": TRAIN_ROW_COUNT, "val": VALIDATION_ROW_COUNT}
    if role not in row_counts or type(row_index) is not int:
        raise MaskedSpatialMetricError("mask role or row index type changed")
    if not 0 <= row_index < row_counts[role]:
        raise MaskedSpatialMetricError("mask row index left its frozen role")
    quadrant_bases = ((0, 0), (0, 8), (8, 0), (8, 8))
    target: set[int] = set()
    for quadrant, (base_row, base_column) in enumerate(quadrant_bases):
        payload = (
            f"{MASK_NAMESPACE}|{role}|{row_index}|{quadrant}"
        ).encode("ascii")
        digest = hashlib.sha256(payload).digest()
        row_offset = int.from_bytes(digest[0:4], "big") % 5
        column_offset = int.from_bytes(digest[4:8], "big") % 5
        for local_row in range(BLOCK_SIZE):
            for local_column in range(BLOCK_SIZE):
                row = base_row + row_offset + local_row
                column = base_column + column_offset + local_column
                target.add(row * GRID_SIZE + column)
    targets = tuple(sorted(target))
    visible = tuple(index for index in range(TOKEN_COUNT) if index not in target)
    if len(targets) != TARGET_TOKEN_COUNT or len(visible) != VISIBLE_TOKEN_COUNT:
        raise MaskedSpatialMetricError("four-quadrant mask cardinality changed")
    return targets, visible


def batched_mask_indices(
    role: str,
    row_indices: Sequence[int],
    *,
    device: Any = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize exact masks as target-Bx64 and visible-Bx192 long tensors."""

    ordered = tuple(row_indices)
    if not ordered:
        raise MaskedSpatialMetricError("a mask batch cannot be empty")
    pairs = tuple(mask_indices(role, index) for index in ordered)
    target = torch.tensor(
        [pair[0] for pair in pairs], dtype=torch.long, device=torch.device(device)
    )
    visible = torch.tensor(
        [pair[1] for pair in pairs], dtype=torch.long, device=torch.device(device)
    )
    return target, visible


def build_validation_donor_indices(rows: Sequence[Any]) -> tuple[int, ...]:
    """Build the frozen same-family, different-scene cyclic donor panel."""

    ordered = tuple(rows)
    if len(ordered) != VALIDATION_ROW_COUNT:
        raise MaskedSpatialMetricError("donors require the complete validation role")
    family_indices = {family: [] for family in REGISTERED_FAMILIES}
    for index, row in enumerate(ordered):
        if (
            getattr(row, "index", None) != index
            or getattr(row, "role", None) != "val"
            or getattr(row, "family", None) not in REGISTERED_FAMILIES
            or type(getattr(row, "scene_id", None)) is not str
            or not row.scene_id
            or type(getattr(row, "current_rgb", None)) is not str
            or not row.current_rgb
        ):
            raise MaskedSpatialMetricError("validation donor metadata changed")
        family_indices[row.family].append(index)
    if any(not family_indices[family] for family in REGISTERED_FAMILIES):
        raise MaskedSpatialMetricError("validation donor panel lost a family")

    donors: list[int] = []
    for row in ordered:
        eligible = [
            candidate
            for candidate in family_indices[row.family]
            if candidate != row.index
            and ordered[candidate].scene_id != row.scene_id
            and ordered[candidate].current_rgb != row.current_rgb
        ]
        if not eligible:
            raise MaskedSpatialMetricError("validation row has no eligible donor")
        donors.append(
            min(
                eligible,
                key=lambda candidate: (
                    (candidate - row.index) % VALIDATION_ROW_COUNT,
                    candidate,
                ),
            )
        )
    return tuple(donors)


def normalize_tokens(value: torch.Tensor) -> torch.Tensor:
    """L2-normalize a finite final-axis token tensor."""

    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.ndim < 2
        or value.shape[-1] < 2
        or not bool(torch.isfinite(value).all())
    ):
        raise MaskedSpatialMetricError("token tensor is invalid")
    return torch.nn.functional.normalize(value, dim=-1, eps=1.0e-8)


def half_squared_token_energy(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return one mean half-squared token energy per row."""

    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[0] < 1
        or prediction.shape[1] != TARGET_TOKEN_COUNT
        or prediction.shape[2] != FEATURE_DIMENSION
        or not prediction.is_floating_point()
        or not target.is_floating_point()
        or not bool(torch.isfinite(prediction).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise MaskedSpatialMetricError("masked energy operands are invalid")
    result = 0.5 * (prediction - target).square().sum(dim=-1).mean(dim=-1)
    if not bool(torch.isfinite(result).all()) or bool((result < 0.0).any()):
        raise MaskedSpatialMetricError("masked energy is invalid")
    return result.detach().to(device="cpu", dtype=torch.float64)


def gather_target_tokens(
    full_tokens: torch.Tensor, target_indices: torch.Tensor
) -> torch.Tensor:
    """Gather Bx64 tokens from a Bx256 raw spatial-token tensor."""

    if (
        not isinstance(full_tokens, torch.Tensor)
        or tuple(full_tokens.shape[1:]) != (TOKEN_COUNT, FEATURE_DIMENSION)
        or not full_tokens.is_floating_point()
        or not isinstance(target_indices, torch.Tensor)
        or target_indices.dtype != torch.long
        or tuple(target_indices.shape)
        != (full_tokens.shape[0], TARGET_TOKEN_COUNT)
        or target_indices.device != full_tokens.device
        or bool((target_indices < 0).any())
        or bool((target_indices >= TOKEN_COUNT).any())
    ):
        raise MaskedSpatialMetricError("masked target gather operands are invalid")
    return full_tokens.gather(
        1, target_indices.unsqueeze(-1).expand(-1, -1, FEATURE_DIMENSION)
    )


def position_mean_targets(full_target_tokens: torch.Tensor) -> torch.Tensor:
    """Mean of per-token normalized EMA targets at each validation position."""

    if (
        not isinstance(full_target_tokens, torch.Tensor)
        or full_target_tokens.ndim != 3
        or tuple(full_target_tokens.shape[1:])
        != (TOKEN_COUNT, FEATURE_DIMENSION)
        or full_target_tokens.shape[0] < 1
    ):
        raise MaskedSpatialMetricError("position-mean source tensor is invalid")
    return normalize_tokens(full_target_tokens).mean(dim=0)


def _energy_vector(value: Any, *, name: str, count: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise MaskedSpatialMetricError(f"{name} must be a floating tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if (
        result.numel() < 1
        or (count is not None and result.numel() != count)
        or not bool(torch.isfinite(result).all())
        or bool((result < 0.0).any())
    ):
        raise MaskedSpatialMetricError(f"{name} energy vector is invalid")
    return result


def _metric_metadata(
    scene_ids: Sequence[str], family_ids: Sequence[str], row_count: int
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    scenes = tuple(scene_ids)
    families = tuple(family_ids)
    if len(scenes) != row_count or len(families) != row_count:
        raise MaskedSpatialMetricError("metric metadata length changed")
    scene_family: dict[str, str] = {}
    for scene, family in zip(scenes, families, strict=True):
        if (
            type(scene) is not str
            or not scene
            or family not in REGISTERED_FAMILIES
        ):
            raise MaskedSpatialMetricError("metric scene or family is invalid")
        previous = scene_family.setdefault(scene, family)
        if previous != family:
            raise MaskedSpatialMetricError("one scene belongs to multiple families")
    if set(scene_family.values()) != set(REGISTERED_FAMILIES):
        raise MaskedSpatialMetricError("metric panel lost a registered family")
    return scenes, scene_family


def _scene_means(
    values: torch.Tensor, scenes: tuple[str, ...], scene_family: Mapping[str, str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for scene in sorted(scene_family):
        indices = [index for index, value in enumerate(scenes) if value == scene]
        result[scene] = float(values[indices].mean())
    return result


def _family_means(
    scene_values: Mapping[str, float], scene_family: Mapping[str, str]
) -> dict[str, float]:
    return {
        family: sum(
            scene_values[scene]
            for scene in scene_values
            if scene_family[scene] == family
        )
        / sum(scene_family[scene] == family for scene in scene_values)
        for family in REGISTERED_FAMILIES
    }


def _scene_family_bootstrap_lower_95(
    advantage_by_scene: Mapping[str, float],
    scene_family: Mapping[str, str],
    *,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    draws = torch.zeros(CONTROL_BOOTSTRAP_REPLICATES, dtype=torch.float64)
    for family in REGISTERED_FAMILIES:
        values = torch.tensor(
            [
                advantage_by_scene[scene]
                for scene in sorted(advantage_by_scene)
                if scene_family[scene] == family
            ],
            dtype=torch.float64,
        )
        if values.numel() < 1:
            raise MaskedSpatialMetricError("bootstrap family has no scene")
        indices = torch.randint(
            values.numel(),
            (CONTROL_BOOTSTRAP_REPLICATES, values.numel()),
            generator=generator,
        )
        draws += values[indices].mean(dim=1)
    draws /= len(REGISTERED_FAMILIES)
    return float(draws.sort().values[CONTROL_BOOTSTRAP_LOWER_INDEX])


def summarize_control(
    correct_energy: torch.Tensor,
    control_energy: torch.Tensor,
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    control_name: str,
) -> ControlSummary:
    """Scene/family-equal summary for one frozen validation control."""

    if control_name not in CONTROL_BOOTSTRAP_SEEDS:
        raise MaskedSpatialMetricError("unregistered control name")
    correct = _energy_vector(correct_energy, name="correct")
    control = _energy_vector(
        control_energy, name=control_name, count=correct.numel()
    )
    scenes, scene_family = _metric_metadata(
        scene_ids, family_ids, int(correct.numel())
    )
    correct_scene = _scene_means(correct, scenes, scene_family)
    control_scene = _scene_means(control, scenes, scene_family)
    advantage_scene = {
        scene: control_scene[scene] - correct_scene[scene]
        for scene in correct_scene
    }
    correct_family = _family_means(correct_scene, scene_family)
    control_family = _family_means(control_scene, scene_family)
    advantage_family = _family_means(advantage_scene, scene_family)
    correct_macro = sum(correct_family.values()) / len(REGISTERED_FAMILIES)
    control_macro = sum(control_family.values()) / len(REGISTERED_FAMILIES)
    if not control_macro > 0.0:
        raise MaskedSpatialMetricError("control macro mean must be positive")
    return ControlSummary(
        correct_macro_mean=correct_macro,
        control_macro_mean=control_macro,
        primary_ratio=correct_macro / max(control_macro, 1.0e-12),
        advantage_macro_mean=(
            sum(advantage_family.values()) / len(REGISTERED_FAMILIES)
        ),
        advantage_bootstrap_lower_95=_scene_family_bootstrap_lower_95(
            advantage_scene,
            scene_family,
            seed=CONTROL_BOOTSTRAP_SEEDS[control_name],
        ),
        positive_family_count=sum(
            value > 0.0 for value in advantage_family.values()
        ),
        correct_by_scene=correct_scene,
        control_by_scene=control_scene,
        advantage_by_scene=advantage_scene,
        advantage_by_family=advantage_family,
    )


class RawHealthAccumulator:
    """Streaming sufficient statistics for the frozen raw-token health metrics."""

    def __init__(self) -> None:
        self.image_count = 0
        self.position_sum = torch.zeros(
            TOKEN_COUNT, FEATURE_DIMENSION, dtype=torch.float64
        )
        self.second_moment = torch.zeros(
            FEATURE_DIMENSION, FEATURE_DIMENSION, dtype=torch.float64
        )
        self.total_square_sum = 0.0
        self.spatial_square_sum = 0.0

    def update(self, tokens: torch.Tensor) -> None:
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 3
            or tuple(tokens.shape[1:]) != (TOKEN_COUNT, FEATURE_DIMENSION)
            or not tokens.is_floating_point()
            or not bool(torch.isfinite(tokens).all())
        ):
            raise MaskedSpatialMetricError("raw health token batch is invalid")
        value = tokens.detach()
        self.image_count += int(value.shape[0])
        self.position_sum += value.sum(dim=0).to(
            device="cpu", dtype=torch.float64
        )
        flat = value.reshape(-1, FEATURE_DIMENSION)
        self.second_moment += (flat.T @ flat).to(
            device="cpu", dtype=torch.float64
        )
        self.total_square_sum += float(value.square().sum())
        within = value - value.mean(dim=1, keepdim=True)
        self.spatial_square_sum += float(within.square().sum())

    def finalize(self) -> RawRepresentationHealth:
        if self.image_count < 1:
            raise MaskedSpatialMetricError("raw health accumulator is empty")
        population = self.image_count * TOKEN_COUNT
        centered_square_sum = (
            self.total_square_sum
            - float(self.position_sum.square().sum()) / self.image_count
        )
        centered_square_sum = max(0.0, centered_square_sum)
        covariance = (
            self.second_moment
            - self.position_sum.T @ self.position_sum / self.image_count
        ) / max(1, population - 1)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        total = eigenvalues.sum()
        if not bool(torch.isfinite(total)) or float(total) <= 0.0:
            effective_rank = 0.0
        else:
            proportions = eigenvalues / total
            entropy = -(
                proportions * proportions.clamp_min(1.0e-12).log()
            ).sum()
            effective_rank = float(entropy.exp())
        result = RawRepresentationHealth(
            image_count=self.image_count,
            token_count=TOKEN_COUNT,
            feature_dimension=FEATURE_DIMENSION,
            effective_rank=effective_rank,
            cross_sample_variance=(
                centered_square_sum
                / (self.image_count * TOKEN_COUNT * FEATURE_DIMENSION)
            ),
            within_image_spatial_diversity=(
                self.spatial_square_sum
                / (self.image_count * TOKEN_COUNT * FEATURE_DIMENSION)
            ),
            finite=True,
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                result.effective_rank,
                result.cross_sample_variance,
                result.within_image_spatial_diversity,
            )
        ):
            raise MaskedSpatialMetricError("raw health result is invalid")
        return result


def raw_representation_health(tokens: torch.Tensor) -> RawRepresentationHealth:
    accumulator = RawHealthAccumulator()
    accumulator.update(tokens)
    return accumulator.finalize()


def raw_health_retention(
    current: RawRepresentationHealth,
    baseline: RawRepresentationHealth,
) -> dict[str, float]:
    if not isinstance(current, RawRepresentationHealth) or not isinstance(
        baseline, RawRepresentationHealth
    ):
        raise TypeError("raw health retention requires health results")
    denominators = (
        baseline.effective_rank,
        baseline.cross_sample_variance,
        baseline.within_image_spatial_diversity,
    )
    if any(not value > 0.0 for value in denominators):
        raise MaskedSpatialMetricError("raw health baseline must be positive")
    return {
        "effective_rank": current.effective_rank / baseline.effective_rank,
        "cross_sample_variance": (
            current.cross_sample_variance / baseline.cross_sample_variance
        ),
        "within_image_spatial_diversity": (
            current.within_image_spatial_diversity
            / baseline.within_image_spatial_diversity
        ),
    }


def flatten_spatial_keys(tokens: torch.Tensor) -> torch.Tensor:
    """Flatten and normalize full raw spatial tokens without fitting a head."""

    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 3
        or tokens.shape[0] < 1
        or tuple(tokens.shape[1:]) != (TOKEN_COUNT, FEATURE_DIMENSION)
        or not tokens.is_floating_point()
        or not bool(torch.isfinite(tokens).all())
    ):
        raise MaskedSpatialMetricError("place raw-token panel is invalid")
    return torch.nn.functional.normalize(
        tokens.detach().reshape(tokens.shape[0], -1), dim=1, eps=1.0e-8
    )


def effective_rank_from_rows(keys: torch.Tensor) -> float:
    """Effective rank via the smaller Gram spectrum for very wide place keys."""

    if (
        not isinstance(keys, torch.Tensor)
        or keys.ndim != 2
        or keys.shape[0] < 2
        or keys.shape[1] < 2
        or not keys.is_floating_point()
        or not bool(torch.isfinite(keys).all())
    ):
        raise MaskedSpatialMetricError("effective-rank key panel is invalid")
    value = keys.detach()
    centered = value - value.mean(dim=0, keepdim=True)
    gram = centered @ centered.T / float(value.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    total = eigenvalues.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        return 0.0
    proportions = eigenvalues / total
    entropy = -(proportions * proportions.clamp_min(1.0e-12).log()).sum()
    result = float(entropy.exp())
    if not math.isfinite(result):
        raise MaskedSpatialMetricError("effective rank is nonfinite")
    return result


def _place_candidate_references(
    rows: Sequence[Any], indices: Sequence[int]
) -> tuple[tuple[str, int, str], ...]:
    references: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for reference_name in ("positive", "negative", "anchor"):
        for index in indices:
            identity = getattr(
                getattr(rows[index], reference_name, None),
                "endpoint_identity_sha256",
                None,
            )
            if type(identity) is not str or not identity:
                raise MaskedSpatialMetricError("place endpoint identity is invalid")
            if identity in seen:
                continue
            seen.add(identity)
            references.append((reference_name, index, identity))
            if len(references) == PLACE_MAXIMUM_CANDIDATES:
                return tuple(references)
    return tuple(references)


def _place_bootstrap_lower_95(
    rows: Sequence[Any], advantages: torch.Tensor
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PLACE_BOOTSTRAP_SEED)
    draws = torch.zeros(CONTROL_BOOTSTRAP_REPLICATES, dtype=torch.float64)
    for family in REGISTERED_FAMILIES:
        indices = [
            index for index, row in enumerate(rows) if row.family == family
        ]
        values = advantages[indices]
        sampled = torch.randint(
            len(indices),
            (CONTROL_BOOTSTRAP_REPLICATES, len(indices)),
            generator=generator,
        )
        draws += values[sampled].mean(dim=1)
    draws /= len(REGISTERED_FAMILIES)
    return float(draws.sort().values[CONTROL_BOOTSTRAP_LOWER_INDEX])


def evaluate_place_keys(
    rows: Sequence[Any],
    online_anchor_keys: torch.Tensor,
    target_anchor_keys: torch.Tensor,
    target_positive_keys: torch.Tensor,
    target_negative_keys: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate the frozen 320-row flattened-token place panel."""

    ordered = tuple(rows)
    panels = (
        online_anchor_keys,
        target_anchor_keys,
        target_positive_keys,
        target_negative_keys,
    )
    if len(ordered) != PLACE_SELECTION_ROW_COUNT or any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[0] != len(ordered)
        or value.shape != panels[0].shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        for value in panels
    ):
        raise MaskedSpatialMetricError("place key panel shape or values changed")
    family_rows = {
        family: [
            index for index, row in enumerate(ordered) if row.family == family
        ]
        for family in REGISTERED_FAMILIES
    }
    if any(
        getattr(row, "index", None) != index
        or getattr(row, "role", None) != "checkpoint_selection"
        or getattr(row, "family", None) not in REGISTERED_FAMILIES
        or type(getattr(row, "scene_id", None)) is not str
        or not row.scene_id
        for index, row in enumerate(ordered)
    ) or any(
        len(family_rows[family]) != PLACE_FAMILY_ROW_COUNTS[family]
        or len({ordered[index].scene_id for index in family_rows[family]}) != 1
        for family in REGISTERED_FAMILIES
    ):
        raise MaskedSpatialMetricError("place panel metadata or quotas changed")

    if len({value.device for value in panels}) != 1:
        raise MaskedSpatialMetricError("place key panels use different devices")
    normalized = tuple(
        torch.nn.functional.normalize(value.detach(), dim=1, eps=1.0e-8)
        for value in panels
    )
    online, target_anchor, target_positive, target_negative = normalized
    positive_energy = 1.0 - (online * target_positive).sum(dim=1)
    negative_energy = 1.0 - (online * target_negative).sum(dim=1)
    advantages = negative_energy - positive_energy
    family_advantage = {
        family: float(advantages[family_rows[family]].mean())
        for family in REGISTERED_FAMILIES
    }

    scene_retrieval: dict[str, dict[str, Any]] = {}
    for family in REGISTERED_FAMILIES:
        indices = family_rows[family]
        scene = ordered[indices[0]].scene_id
        references = _place_candidate_references(ordered, indices)
        if not PLACE_MINIMUM_CANDIDATES <= len(references) <= PLACE_MAXIMUM_CANDIDATES:
            raise MaskedSpatialMetricError("place candidate count left [32,64]")
        panel_by_name = {
            "anchor": target_anchor,
            "positive": target_positive,
            "negative": target_negative,
        }
        candidate_ids = [identity for _name, _index, identity in references]
        candidate_position = {
            identity: index for index, identity in enumerate(candidate_ids)
        }
        targets = torch.stack(
            [panel_by_name[name][index] for name, index, _identity in references]
        )
        energy = 1.0 - online[indices] @ targets.T
        ranks: list[int] = []
        for local_index, row_index in enumerate(indices):
            positive_identity = ordered[
                row_index
            ].positive.endpoint_identity_sha256
            if positive_identity not in candidate_position:
                raise MaskedSpatialMetricError(
                    "paired positive is absent from place candidates"
                )
            relevant = candidate_position[positive_identity]
            ranks.append(
                int((energy[local_index] <= energy[local_index, relevant]).sum())
            )
        recall = sum(rank <= 5 for rank in ranks) / len(ranks)
        chance = 5.0 / len(references)
        scene_retrieval[scene] = {
            "family": family,
            "query_count": len(indices),
            "candidate_count": len(references),
            "recall_at_5": recall,
            "exact_chance_recall_at_5": chance,
            "chance_multiple": recall / chance,
            "mean_pessimistic_rank": sum(ranks) / len(ranks),
        }
    recall = sum(
        value["recall_at_5"] for value in scene_retrieval.values()
    ) / len(scene_retrieval)
    chance = sum(
        value["exact_chance_recall_at_5"]
        for value in scene_retrieval.values()
    ) / len(scene_retrieval)
    scene_equal_chance_multiple = sum(
        value["chance_multiple"] for value in scene_retrieval.values()
    ) / len(scene_retrieval)
    result = {
        "row_count": len(ordered),
        "scene_count": len(scene_retrieval),
        "energy": {
            "positive_macro_mean": sum(
                float(positive_energy[family_rows[family]].mean())
                for family in REGISTERED_FAMILIES
            )
            / len(REGISTERED_FAMILIES),
            "negative_macro_mean": sum(
                float(negative_energy[family_rows[family]].mean())
                for family in REGISTERED_FAMILIES
            )
            / len(REGISTERED_FAMILIES),
            "negative_minus_positive_macro_mean": (
                sum(family_advantage.values()) / len(REGISTERED_FAMILIES)
            ),
            "negative_minus_positive_bootstrap_lower_95":
                _place_bootstrap_lower_95(
                    ordered, advantages.to(device="cpu", dtype=torch.float64)
                ),
            "positive_family_count": sum(
                value > 0.0 for value in family_advantage.values()
            ),
            "negative_minus_positive_by_family": family_advantage,
        },
        "retrieval": {
            "recall_at_5": recall,
            "exact_chance_recall_at_5": chance,
            "chance_multiple": scene_equal_chance_multiple,
            "aggregate_recall_over_aggregate_chance": recall / chance,
            "scene_count_above_chance": sum(
                value["recall_at_5"] > value["exact_chance_recall_at_5"]
                for value in scene_retrieval.values()
            ),
            "scene_count_at_least_1_5x_chance": sum(
                value["recall_at_5"]
                >= 1.5 * value["exact_chance_recall_at_5"]
                for value in scene_retrieval.values()
            ),
            "by_scene": scene_retrieval,
        },
        "target_place_key_effective_rank": effective_rank_from_rows(
            target_positive
        ),
    }
    return result


def place_retention(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    current_multiple = float(current["retrieval"]["chance_multiple"])
    baseline_multiple = float(baseline["retrieval"]["chance_multiple"])
    current_rank = float(current["target_place_key_effective_rank"])
    baseline_rank = float(baseline["target_place_key_effective_rank"])
    if not baseline_multiple > 0.0 or not baseline_rank > 0.0:
        raise MaskedSpatialMetricError("place baseline must be positive")
    return {
        "chance_multiple": current_multiple / baseline_multiple,
        "target_place_key_effective_rank": current_rank / baseline_rank,
    }


__all__ = [
    "BLOCK_SIZE",
    "CONTROL_BOOTSTRAP_REPLICATES",
    "CONTROL_BOOTSTRAP_SEEDS",
    "ControlSummary",
    "FEATURE_DIMENSION",
    "GRID_SIZE",
    "MASK_NAMESPACE",
    "MaskedSpatialMetricError",
    "OBSERVATION_UPDATES",
    "PLACE_BOOTSTRAP_SEED",
    "REGISTERED_FAMILIES",
    "RawHealthAccumulator",
    "RawRepresentationHealth",
    "TARGET_TOKEN_COUNT",
    "TOKEN_COUNT",
    "TRAIN_ROW_COUNT",
    "VALIDATION_ROW_COUNT",
    "VISIBLE_TOKEN_COUNT",
    "batched_mask_indices",
    "build_validation_donor_indices",
    "effective_rank_from_rows",
    "evaluate_place_keys",
    "flatten_spatial_keys",
    "gather_target_tokens",
    "half_squared_token_energy",
    "mask_indices",
    "normalize_tokens",
    "place_retention",
    "position_mean_targets",
    "raw_health_retention",
    "raw_representation_health",
    "summarize_control",
]
