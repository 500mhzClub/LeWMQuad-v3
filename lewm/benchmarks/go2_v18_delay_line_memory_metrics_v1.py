"""Pure metrics and frozen gates for the V18 delay-line memory JEPA V1.

The module accepts already-computed energy tensors and public observation
metadata.  It performs no model construction, data access, filesystem access,
or device discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


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
HORIZON_COUNT = 4
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_731
PARTICIPATION_RANK_RATIO_MIN = 0.10
NEAR_ZERO_STD = 0.02
NEAR_ZERO_FRACTION_MAX = 0.05


@dataclass(frozen=True)
class MacroMetric:
    """One four-horizon metric after scene-then-family aggregation."""

    per_scene: Mapping[str, tuple[float, ...]]
    per_family: Mapping[str, tuple[float, ...]]
    macro: tuple[float, ...]
    bootstrap_lower_95: tuple[float, ...]
    positive_family_count: tuple[int, ...]


@dataclass(frozen=True)
class TemporalMetrics:
    """Normalized temporal metrics from the six registered energy arms."""

    row_count: int
    scene_count: int
    family_count: int
    bootstrap_replicates: int
    bootstrap_seed: int
    score: MacroMetric
    persistence_lift: MacroMetric
    action_lift: MacroMetric
    history_lift: MacroMetric


@dataclass(frozen=True)
class ParticipationRank:
    """Scale and participation-rank audit for rows of learned state."""

    row_count: int
    feature_dimension: int
    effective_rank: float
    participation_rank_ratio: float
    near_zero_fraction: float
    rms: float
    finite: bool
    nonzero_scale: bool
    noncollapsed: bool


@dataclass(frozen=True)
class RuntimeSafeguards:
    """Integrity facts supplied by the future reviewed runner."""

    integrity_pass: bool
    perception_safeguards_pass: bool
    gradient_accounting_pass: bool
    target_noncollapsed: bool
    online_noncollapsed: bool


@dataclass(frozen=True)
class SubstrateMetrics:
    """Perception metrics needed by the frozen continuation/terminal gates."""

    place_chance_multiple: float
    place_scene_count_above_chance: int
    target_place_rank: float
    target_place_rank_retention: float
    physical_passed_margin_count: int
    physical_causal_control_pass_count: int


@dataclass(frozen=True)
class ObservationMetrics:
    """All pure metric inputs for one registered observation."""

    update: int
    temporal: TemporalMetrics
    memory_state: ParticipationRank
    safeguards: RuntimeSafeguards
    substrate: SubstrateMetrics


@dataclass(frozen=True)
class GateDecision:
    update: int
    status: str
    action: str
    passed: bool
    checks: Mapping[str, bool]
    failed_checks: tuple[str, ...]
    observed: Mapping[str, float | int | bool | None]
    selected_update: int | None = None


def _energy_tensor(
    value: torch.Tensor,
    *,
    name: str,
    expected_shape: tuple[int, int] | None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating torch.Tensor")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != HORIZON_COUNT:
        raise ValueError(f"{name} must have shape (N,{HORIZON_COUNT})")
    shape = (int(value.shape[0]), int(value.shape[1]))
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"{name} shape differs from real_energy")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(result).all()) or bool((result < 0.0).any()):
        raise ValueError(f"{name} must contain finite nonnegative energies")
    return result


def _metadata(
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    row_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...], Mapping[str, str]]:
    scenes = tuple(scene_ids)
    families = tuple(family_ids)
    if len(scenes) != row_count or len(families) != row_count:
        raise ValueError("scene_ids and family_ids must match the energy row count")
    if any(type(value) is not str or not value for value in scenes + families):
        raise ValueError("scene and family identifiers must be nonempty strings")
    if set(families) != set(REGISTERED_FAMILIES):
        raise ValueError("evaluation must contain exactly the eight registered families")
    scene_family: dict[str, str] = {}
    for scene, family in zip(scenes, families, strict=True):
        if family not in REGISTERED_FAMILIES:
            raise ValueError(f"unregistered family {family!r}")
        previous = scene_family.setdefault(scene, family)
        if previous != family:
            raise ValueError("one scene cannot belong to multiple families")
    return scenes, families, scene_family


def _macro_metrics(
    row_values: torch.Tensor,
    scenes: tuple[str, ...],
    scene_family: Mapping[str, str],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[MacroMetric, ...]:
    # row_values: (N, metric=4, horizon=4)
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 40:
        raise ValueError("bootstrap_replicates must be an integer at least 40")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")

    scene_names = tuple(sorted(scene_family))
    per_scene_tensor: dict[str, torch.Tensor] = {}
    for scene in scene_names:
        indices = [index for index, value in enumerate(scenes) if value == scene]
        per_scene_tensor[scene] = row_values[indices].mean(dim=0)

    family_tensors: dict[str, torch.Tensor] = {}
    family_scenes: dict[str, tuple[str, ...]] = {}
    for family in REGISTERED_FAMILIES:
        names = tuple(
            scene for scene in scene_names if scene_family[scene] == family
        )
        if not names:
            raise ValueError(f"family {family!r} has no evaluation scene")
        family_scenes[family] = names
        family_tensors[family] = torch.stack(
            [per_scene_tensor[scene] for scene in names]
        ).mean(dim=0)
    macro = torch.stack(tuple(family_tensors.values())).mean(dim=0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(bootstrap_seed)
    bootstrap = torch.zeros(
        bootstrap_replicates,
        row_values.shape[1],
        row_values.shape[2],
        dtype=torch.float64,
    )
    for family in REGISTERED_FAMILIES:
        values = torch.stack(
            [per_scene_tensor[scene] for scene in family_scenes[family]]
        )
        scene_count = values.shape[0]
        indices = torch.randint(
            scene_count,
            (bootstrap_replicates, scene_count),
            generator=generator,
        )
        bootstrap += values[indices].mean(dim=1)
    bootstrap /= len(REGISTERED_FAMILIES)
    lower_index = int(0.025 * bootstrap_replicates)
    lower = bootstrap.sort(dim=0).values[lower_index]

    summaries: list[MacroMetric] = []
    for metric_index in range(row_values.shape[1]):
        per_scene = {
            scene: tuple(
                float(value)
                for value in per_scene_tensor[scene][metric_index].tolist()
            )
            for scene in scene_names
        }
        per_family = {
            family: tuple(
                float(value)
                for value in family_tensors[family][metric_index].tolist()
            )
            for family in REGISTERED_FAMILIES
        }
        summaries.append(
            MacroMetric(
                per_scene=per_scene,
                per_family=per_family,
                macro=tuple(float(value) for value in macro[metric_index].tolist()),
                bootstrap_lower_95=tuple(
                    float(value) for value in lower[metric_index].tolist()
                ),
                positive_family_count=tuple(
                    sum(per_family[family][horizon] > 0.0 for family in per_family)
                    for horizon in range(HORIZON_COUNT)
                ),
            )
        )
    return tuple(summaries)


def evaluate_temporal_metrics(
    real_energy: torch.Tensor,
    persistence_energy: torch.Tensor,
    wrong_action_energy: torch.Tensor,
    reset_energy: torch.Tensor,
    reverse_energy: torch.Tensor,
    shuffle_energy: torch.Tensor,
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> TemporalMetrics:
    """Compute the frozen S/P/A/H metrics and stratified scene bootstrap."""

    real = _energy_tensor(real_energy, name="real_energy", expected_shape=None)
    shape = (int(real.shape[0]), int(real.shape[1]))
    persistence = _energy_tensor(
        persistence_energy, name="persistence_energy", expected_shape=shape
    )
    wrong = _energy_tensor(
        wrong_action_energy, name="wrong_action_energy", expected_shape=shape
    )
    reset = _energy_tensor(reset_energy, name="reset_energy", expected_shape=shape)
    reverse = _energy_tensor(
        reverse_energy, name="reverse_energy", expected_shape=shape
    )
    shuffle = _energy_tensor(
        shuffle_energy, name="shuffle_energy", expected_shape=shape
    )
    if bool((persistence <= 0.0).any()):
        raise ValueError("persistence_energy must be strictly positive")
    scenes, _families, scene_family = _metadata(
        scene_ids, family_ids, row_count=shape[0]
    )

    score = real / persistence
    persistence_lift = 1.0 - score
    action_lift = (wrong - real) / persistence
    best_corrupted_history = torch.minimum(torch.minimum(reset, reverse), shuffle)
    history_lift = (best_corrupted_history - real) / persistence
    values = torch.stack(
        (score, persistence_lift, action_lift, history_lift), dim=1
    )
    summaries = _macro_metrics(
        values,
        scenes,
        scene_family,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return TemporalMetrics(
        row_count=shape[0],
        scene_count=len(scene_family),
        family_count=len(REGISTERED_FAMILIES),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        score=summaries[0],
        persistence_lift=summaries[1],
        action_lift=summaries[2],
        history_lift=summaries[3],
    )


def participation_rank(
    feature_rows: torch.Tensor,
    *,
    near_zero_std: float = NEAR_ZERO_STD,
) -> ParticipationRank:
    """Audit the final-axis feature geometry without retaining input tensors."""

    if not isinstance(feature_rows, torch.Tensor) or not feature_rows.is_floating_point():
        raise TypeError("feature_rows must be a floating torch.Tensor")
    if feature_rows.ndim < 2 or feature_rows.shape[-1] < 2:
        raise ValueError("feature_rows must have at least two rows and features")
    value = feature_rows.detach().to(device="cpu", dtype=torch.float64).reshape(
        -1, feature_rows.shape[-1]
    )
    if value.shape[0] < 2:
        raise ValueError("feature_rows must have at least two rows and features")
    if not math.isfinite(near_zero_std) or near_zero_std <= 0.0:
        raise ValueError("near_zero_std must be finite and positive")
    finite = bool(torch.isfinite(value).all())
    if not finite:
        return ParticipationRank(
            row_count=int(value.shape[0]),
            feature_dimension=int(value.shape[1]),
            effective_rank=0.0,
            participation_rank_ratio=0.0,
            near_zero_fraction=1.0,
            rms=float("nan"),
            finite=False,
            nonzero_scale=False,
            noncollapsed=False,
        )

    rms = float(value.square().mean().sqrt().item())
    centered = value - value.mean(dim=0, keepdim=True)
    std = centered.var(dim=0, unbiased=False).sqrt()
    near_zero_fraction = float((std < near_zero_std).to(torch.float64).mean())
    covariance = centered.T @ centered / max(1, value.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if float(total) <= 0.0 or not bool(torch.isfinite(total)):
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / total
        entropy = -(
            probabilities * probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()
        ).sum()
        effective_rank = float(entropy.exp().item())
    ratio = effective_rank / int(value.shape[1])
    nonzero_scale = math.isfinite(rms) and rms > 0.0
    noncollapsed = (
        finite
        and nonzero_scale
        and ratio >= PARTICIPATION_RANK_RATIO_MIN
        and near_zero_fraction <= NEAR_ZERO_FRACTION_MAX
    )
    return ParticipationRank(
        row_count=int(value.shape[0]),
        feature_dimension=int(value.shape[1]),
        effective_rank=effective_rank,
        participation_rank_ratio=ratio,
        near_zero_fraction=near_zero_fraction,
        rms=rms,
        finite=finite,
        nonzero_scale=nonzero_scale,
        noncollapsed=noncollapsed,
    )


def _validate_observation(observation: ObservationMetrics, *, update: int) -> None:
    if not isinstance(observation, ObservationMetrics) or observation.update != update:
        raise ValueError(f"expected registered update {update} observation")
    if observation.temporal.family_count != len(REGISTERED_FAMILIES):
        raise ValueError("temporal observation lost a registered family")
    for value in observation.safeguards.__dict__.values():
        if type(value) is not bool:
            raise TypeError("runtime safeguard fields must be exact booleans")
    substrate = observation.substrate
    floating = (
        substrate.place_chance_multiple,
        substrate.target_place_rank,
        substrate.target_place_rank_retention,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in floating):
        raise ValueError("substrate floating metrics must be finite and nonnegative")
    if (
        type(substrate.place_scene_count_above_chance) is not int
        or not 0 <= substrate.place_scene_count_above_chance <= 8
        or type(substrate.physical_passed_margin_count) is not int
        or not 0 <= substrate.physical_passed_margin_count <= 189
        or type(substrate.physical_causal_control_pass_count) is not int
        or not 0 <= substrate.physical_causal_control_pass_count <= 12
    ):
        raise ValueError("substrate count metric is out of range")


def _decision(
    *,
    update: int,
    status: str,
    action: str,
    checks: Mapping[str, bool],
    observed: Mapping[str, float | int | bool | None],
    selected_update: int | None = None,
) -> GateDecision:
    frozen_checks = dict(checks)
    failed = tuple(name for name, passed in frozen_checks.items() if not passed)
    return GateDecision(
        update=update,
        status=status,
        action=action,
        passed=not failed,
        checks=frozen_checks,
        failed_checks=failed,
        observed=dict(observed),
        selected_update=selected_update,
    )


def update250_futility_decision(
    update100: ObservationMetrics,
    update250: ObservationMetrics,
) -> GateDecision:
    """Apply the exact update-250 collapse and joint-futility stop."""

    _validate_observation(update100, update=100)
    _validate_observation(update250, update=250)
    structural_checks = {
        "integrity_pass": update250.safeguards.integrity_pass,
        "target_noncollapsed": update250.safeguards.target_noncollapsed,
        "online_noncollapsed": update250.safeguards.online_noncollapsed,
        "memory_noncollapsed": update250.memory_state.noncollapsed,
        "place_at_least_1p5x_chance": (
            update250.substrate.place_chance_multiple >= 1.5
        ),
        "gradient_accounting_pass": (
            update250.safeguards.gradient_accounting_pass
        ),
    }

    def futile(observation: ObservationMetrics) -> bool:
        return (
            observation.temporal.persistence_lift.macro[3] <= 0.0
            and observation.temporal.action_lift.macro[3] <= 0.0
            and observation.temporal.history_lift.macro[3] <= 0.0
            and observation.temporal.history_lift.positive_family_count[3] == 0
        )

    jointly_futile = futile(update100) and futile(update250)
    checks = {
        **structural_checks,
        "not_jointly_futile_at_updates_100_and_250": not jointly_futile,
    }
    observed = {
        "update100_h4_persistence_lift": (
            update100.temporal.persistence_lift.macro[3]
        ),
        "update100_h4_action_lift": update100.temporal.action_lift.macro[3],
        "update100_h4_history_lift": update100.temporal.history_lift.macro[3],
        "update250_h4_persistence_lift": (
            update250.temporal.persistence_lift.macro[3]
        ),
        "update250_h4_action_lift": update250.temporal.action_lift.macro[3],
        "update250_h4_history_lift": update250.temporal.history_lift.macro[3],
        "update250_history_positive_family_count": (
            update250.temporal.history_lift.positive_family_count[3]
        ),
        "memory_participation_rank_ratio": (
            update250.memory_state.participation_rank_ratio
        ),
        "memory_near_zero_fraction": update250.memory_state.near_zero_fraction,
        "place_chance_multiple": update250.substrate.place_chance_multiple,
    }
    if not all(structural_checks.values()):
        return _decision(
            update=250,
            status="STOP_UPDATE250_INTEGRITY_OR_COLLAPSE",
            action="STOP_TERMINAL",
            checks=checks,
            observed=observed,
        )
    return _decision(
        update=250,
        status=(
            "STOP_UPDATE250_SCIENTIFIC_FUTILITY"
            if jointly_futile
            else "CONTINUE_TO_UPDATE500"
        ),
        action="STOP_TERMINAL" if jointly_futile else "CONTINUE",
        checks=checks,
        observed=observed,
    )


def update500_continuation_decision(
    update500: ObservationMetrics,
) -> GateDecision:
    """Apply the frozen Stage-A continuation gate."""

    _validate_observation(update500, update=500)
    temporal = update500.temporal
    checks = {
        "integrity_pass": update500.safeguards.integrity_pass,
        "perception_safeguards_pass": (
            update500.safeguards.perception_safeguards_pass
        ),
        "gradient_accounting_pass": (
            update500.safeguards.gradient_accounting_pass
        ),
        "target_noncollapsed": update500.safeguards.target_noncollapsed,
        "online_noncollapsed": update500.safeguards.online_noncollapsed,
        "memory_noncollapsed": update500.memory_state.noncollapsed,
        "h4_persistence_lift_positive": temporal.persistence_lift.macro[3] > 0.0,
        "h4_persistence_lower_95_positive": (
            temporal.persistence_lift.bootstrap_lower_95[3] > 0.0
        ),
        "persistence_positive_in_six_families": (
            temporal.persistence_lift.positive_family_count[3] >= 6
        ),
        "h4_action_lift_positive": temporal.action_lift.macro[3] > 0.0,
        "h4_action_lower_95_positive": (
            temporal.action_lift.bootstrap_lower_95[3] > 0.0
        ),
        "action_positive_in_six_families": (
            temporal.action_lift.positive_family_count[3] >= 6
        ),
        "history_positive_in_four_families": (
            temporal.history_lift.positive_family_count[3] >= 4
        ),
        "mean_h1_h4_persistence_lift_positive": (
            math.fsum(temporal.persistence_lift.macro) / HORIZON_COUNT > 0.0
        ),
    }
    passed = all(checks.values())
    return _decision(
        update=500,
        status=(
            "CONTINUE_TO_UPDATE1000"
            if passed
            else "STOP_UPDATE500_CONTINUATION_GATE"
        ),
        action="CONTINUE" if passed else "STOP_TERMINAL",
        checks=checks,
        observed={
            "h4_persistence_lift": temporal.persistence_lift.macro[3],
            "h4_persistence_lower_95": (
                temporal.persistence_lift.bootstrap_lower_95[3]
            ),
            "h4_persistence_positive_family_count": (
                temporal.persistence_lift.positive_family_count[3]
            ),
            "h4_action_lift": temporal.action_lift.macro[3],
            "h4_action_lower_95": temporal.action_lift.bootstrap_lower_95[3],
            "h4_action_positive_family_count": (
                temporal.action_lift.positive_family_count[3]
            ),
            "h4_history_lift": temporal.history_lift.macro[3],
            "h4_history_positive_family_count": (
                temporal.history_lift.positive_family_count[3]
            ),
            "mean_h1_h4_persistence_lift": (
                math.fsum(temporal.persistence_lift.macro) / HORIZON_COUNT
            ),
        },
    )


def _terminal_eligible(observation: ObservationMetrics) -> bool:
    temporal = observation.temporal
    safeguards = observation.safeguards
    return (
        safeguards.integrity_pass
        and safeguards.perception_safeguards_pass
        and safeguards.gradient_accounting_pass
        and safeguards.target_noncollapsed
        and safeguards.online_noncollapsed
        and observation.memory_state.noncollapsed
        and temporal.persistence_lift.macro[3] > 0.0
        and temporal.action_lift.macro[3] > 0.0
        and temporal.history_lift.macro[3] > 0.0
    )


def terminal_qualification_decision(
    observations: Sequence[ObservationMetrics],
) -> GateDecision:
    """Select the registered eligible observation, then apply terminal gates."""

    by_update = {observation.update: observation for observation in observations}
    if len(by_update) != len(tuple(observations)) or set(by_update) != {500, 750, 1000}:
        raise ValueError("terminal selection requires unique updates 500, 750, and 1000")
    for update in (500, 750, 1000):
        _validate_observation(by_update[update], update=update)
    eligible = [
        observation
        for observation in by_update.values()
        if _terminal_eligible(observation)
    ]
    if not eligible:
        return GateDecision(
            update=1000,
            status="FAIL_TERMINAL_NO_ELIGIBLE_OBSERVATION",
            action="STOP_TERMINAL",
            passed=False,
            checks={"eligible_observation_exists": False},
            failed_checks=("eligible_observation_exists",),
            observed={"eligible_observation_count": 0, "selected_mean_score": None},
            selected_update=None,
        )
    selected = min(
        eligible,
        key=lambda observation: (
            math.fsum(observation.temporal.score.macro) / HORIZON_COUNT,
            observation.update,
        ),
    )
    temporal = selected.temporal
    substrate = selected.substrate
    memory = selected.memory_state
    history_nonnegative_horizons = sum(
        value >= 0.0 for value in temporal.history_lift.macro
    )
    checks = {
        "eligible_observation_exists": True,
        "positive_persistence_lift_h1_h4": all(
            value > 0.0 for value in temporal.persistence_lift.macro
        ),
        "h4_persistence_lift_at_least_0p10": (
            temporal.persistence_lift.macro[3] >= 0.10
        ),
        "h4_persistence_lower_95_positive": (
            temporal.persistence_lift.bootstrap_lower_95[3] > 0.0
        ),
        "persistence_positive_in_six_families": (
            temporal.persistence_lift.positive_family_count[3] >= 6
        ),
        "h4_action_lift_at_least_0p05": (
            temporal.action_lift.macro[3] >= 0.05
        ),
        "h4_action_lower_95_positive": (
            temporal.action_lift.bootstrap_lower_95[3] > 0.0
        ),
        "action_positive_in_six_families": (
            temporal.action_lift.positive_family_count[3] >= 6
        ),
        "h4_history_lift_at_least_0p03": (
            temporal.history_lift.macro[3] >= 0.03
        ),
        "h4_history_lower_95_positive": (
            temporal.history_lift.bootstrap_lower_95[3] > 0.0
        ),
        "history_positive_in_six_families": (
            temporal.history_lift.positive_family_count[3] >= 6
        ),
        "history_nonnegative_at_three_horizons": (
            history_nonnegative_horizons >= 3
        ),
        "memory_participation_rank_ratio_at_least_0p10": (
            memory.participation_rank_ratio >= PARTICIPATION_RANK_RATIO_MIN
        ),
        "memory_state_finite": memory.finite,
        "memory_state_nonzero_scale": memory.nonzero_scale,
        "memory_near_zero_fraction_at_most_0p05": (
            memory.near_zero_fraction <= NEAR_ZERO_FRACTION_MAX
        ),
        "target_perception_noncollapsed": selected.safeguards.target_noncollapsed,
        "online_perception_noncollapsed": selected.safeguards.online_noncollapsed,
        "place_at_least_2x_chance": substrate.place_chance_multiple >= 2.0,
        "place_above_chance_in_six_scenes": (
            substrate.place_scene_count_above_chance >= 6
        ),
        "target_place_rank_at_least_2": substrate.target_place_rank >= 2.0,
        "target_place_rank_retains_80_percent": (
            substrate.target_place_rank_retention >= 0.80
        ),
        "physical_margins_at_least_60_of_189": (
            substrate.physical_passed_margin_count >= 60
        ),
        "all_12_physical_causal_controls_pass": (
            substrate.physical_causal_control_pass_count == 12
        ),
    }
    passed = all(checks.values())
    return _decision(
        update=1000,
        status=(
            "PASS_SHORT_HORIZON_CAUSAL_MEMORY_SUBSTRATE"
            if passed
            else "FAIL_TERMINAL_QUALIFICATION"
        ),
        action="QUALIFY_SHORT_HORIZON_MEMORY" if passed else "STOP_TERMINAL",
        checks=checks,
        observed={
            "eligible_observation_count": len(eligible),
            "selected_mean_score": (
                math.fsum(temporal.score.macro) / HORIZON_COUNT
            ),
            "h4_persistence_lift": temporal.persistence_lift.macro[3],
            "h4_action_lift": temporal.action_lift.macro[3],
            "h4_history_lift": temporal.history_lift.macro[3],
            "history_nonnegative_horizon_count": history_nonnegative_horizons,
            "memory_participation_rank_ratio": memory.participation_rank_ratio,
            "memory_near_zero_fraction": memory.near_zero_fraction,
            "place_chance_multiple": substrate.place_chance_multiple,
            "place_scene_count_above_chance": (
                substrate.place_scene_count_above_chance
            ),
            "target_place_rank": substrate.target_place_rank,
            "target_place_rank_retention": substrate.target_place_rank_retention,
            "physical_passed_margin_count": (
                substrate.physical_passed_margin_count
            ),
            "physical_causal_control_pass_count": (
                substrate.physical_causal_control_pass_count
            ),
        },
        selected_update=selected.update,
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "GateDecision",
    "HORIZON_COUNT",
    "MacroMetric",
    "NEAR_ZERO_FRACTION_MAX",
    "NEAR_ZERO_STD",
    "ObservationMetrics",
    "PARTICIPATION_RANK_RATIO_MIN",
    "ParticipationRank",
    "REGISTERED_FAMILIES",
    "RuntimeSafeguards",
    "SubstrateMetrics",
    "TemporalMetrics",
    "evaluate_temporal_metrics",
    "participation_rank",
    "terminal_qualification_decision",
    "update250_futility_decision",
    "update500_continuation_decision",
]
