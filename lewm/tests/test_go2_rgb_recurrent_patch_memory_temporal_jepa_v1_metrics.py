from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch

from lewm.benchmarks import (
    go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as metrics,
)


def _rows(
    role: str,
    *,
    scenes_per_family: int,
    rows_per_scene: int,
) -> tuple[metrics.MetadataRow, ...]:
    result = []
    for family_index, family in enumerate(metrics.REGISTERED_FAMILIES):
        for scene_index in range(scenes_per_family):
            scene = f"{family}_{family_index:02x}{scene_index:010x}"
            for within_scene in range(rows_per_scene):
                index = len(result)
                result.append(
                    metrics.MetadataRow(
                        index=index,
                        role=role,
                        family=family,
                        scene_id=scene,
                        rgb=tuple(f"{scene}/rgb/{index}-{step}" for step in range(7)),
                        actions=tuple(
                            (index + position) % metrics.ACTION_COUNT
                            for position in range(6)
                        ),
                    )
                )
    return tuple(result)


def _independent_hash_order(namespace: str, values: list[int]) -> tuple[int, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                hashlib.sha256(f"{namespace}/{value}".encode("ascii")).digest(),
                value,
            ),
        )
    )


def test_schedule_and_sentinel_follow_exact_scene_round_robin_then_hash_order() -> None:
    train = _rows("train", scenes_per_family=3, rows_per_scene=2)
    schedule = metrics.build_training_schedule(train, rows_per_family=5)
    expected_train = []
    for family in metrics.REGISTERED_FAMILIES:
        family_rows = [row for row in train if row.family == family]
        scenes = sorted({row.scene_id for row in family_rows})
        expected_train.extend(
            next(row.index for row in family_rows if row.scene_id == scene)
            for scene in scenes
        )
        expected_train.extend(
            next(
                row.index
                for row in family_rows
                if row.scene_id == scene
                and row.index
                != next(
                    first.index
                    for first in family_rows
                    if first.scene_id == scene
                )
            )
            for scene in scenes[:2]
        )
    assert schedule == _independent_hash_order(
        metrics.TRAIN_SCHEDULE_NAMESPACE, expected_train
    )
    assert len(schedule) == 40
    assert len(set(schedule)) == 40

    validation = _rows("val", scenes_per_family=2, rows_per_scene=3)
    sentinel = metrics.build_sentinel_indices(validation, rows_per_family=3)
    expected_sentinel = []
    for family in metrics.REGISTERED_FAMILIES:
        family_rows = [row for row in validation if row.family == family]
        scenes = sorted({row.scene_id for row in family_rows})
        first = [
            next(row.index for row in family_rows if row.scene_id == scene)
            for scene in scenes
        ]
        second_first_scene = next(
            row.index
            for row in family_rows
            if row.scene_id == scenes[0] and row.index != first[0]
        )
        expected_sentinel.extend((*first, second_first_scene))
    assert sentinel == _independent_hash_order(
        metrics.SENTINEL_NAMESPACE, expected_sentinel
    )


def test_wrong_history_donors_masks_and_panel_identity_are_frozen() -> None:
    rows = _rows("val", scenes_per_family=2, rows_per_scene=3)
    sentinel = metrics.build_sentinel_indices(rows, rows_per_family=3)
    donors = metrics.build_wrong_history_donor_indices(
        rows, selected_indices=sentinel
    )
    for row_index, observed in zip(sentinel, donors, strict=True):
        row = rows[row_index]
        eligible = [
            candidate.index
            for candidate in rows
            if candidate.family == row.family
            and candidate.scene_id != row.scene_id
        ]
        expected = min(
            eligible,
            key=lambda donor: (
                hashlib.sha256(
                    (
                        f"{metrics.WRONG_HISTORY_NAMESPACE}/"
                        f"{row_index}/{donor}"
                    ).encode("ascii")
                ).digest(),
                donor,
            ),
        )
        assert observed == expected

    target, visible = metrics.mask_indices("val", 17)
    independent = []
    for quadrant, (base_row, base_column) in enumerate(
        ((0, 0), (0, 8), (8, 0), (8, 8))
    ):
        digest = hashlib.sha256(
            f"{metrics.MASK_NAMESPACE}|val|17|{quadrant}".encode("ascii")
        ).digest()
        row_offset = int.from_bytes(digest[:4], "big") % 5
        column_offset = int.from_bytes(digest[4:8], "big") % 5
        independent.extend(
            (base_row + row_offset + dr) * 16
            + base_column
            + column_offset
            + dc
            for dr in range(4)
            for dc in range(4)
        )
    assert target == tuple(sorted(independent))
    assert len(target) == 64 and len(visible) == 192
    batched_target, batched_visible = metrics.batched_mask_indices(
        "val", (17, 18), device="cpu"
    )
    assert tuple(batched_target.shape) == (2, 64)
    assert tuple(batched_visible.shape) == (2, 192)

    wrong_action = metrics.wrong_action_eligible_indices(
        rows, selected_indices=sentinel
    )
    identity = metrics.validation_panel_identity(
        rows, sentinel, donors, wrong_action
    )
    assert identity == metrics.validation_panel_identity(
        rows, sentinel, donors, wrong_action
    )
    assert len(identity) == 64


def test_control_summary_is_scene_family_equal_and_seeded() -> None:
    scenes = []
    families = []
    real = []
    control = []
    for family_index, family in enumerate(metrics.REGISTERED_FAMILIES):
        for scene_index in range(2):
            for _ in range(scene_index + 1):
                scenes.append(f"{family}-{scene_index}")
                families.append(family)
                real.append(1.0 + family_index * 0.01)
                control.append(2.0 + family_index * 0.01)
    observed = metrics.summarize_control(
        torch.tensor(real),
        torch.tensor(control),
        scenes,
        families,
        control_name="persistence",
    )
    assert observed.correct_macro_mean == pytest.approx(1.035)
    assert observed.control_macro_mean == pytest.approx(2.035)
    assert observed.primary_ratio == pytest.approx(1.035 / 2.035)
    assert observed.advantage_macro_mean == pytest.approx(1.0)
    assert observed.advantage_bootstrap_lower_95 == pytest.approx(1.0)
    assert observed.positive_family_count == 8
    assert observed == metrics.summarize_control(
        torch.tensor(real),
        torch.tensor(control),
        scenes,
        families,
        control_name="persistence",
    )


def test_energy_health_and_accounting_match_registered_formulas() -> None:
    prediction = torch.zeros(2, 64, 192)
    target = torch.zeros_like(prediction)
    prediction[:, :, 0] = 1.0
    target[:, :, 1] = 1.0
    energy = metrics.normalized_half_squared_energy(prediction, target)
    assert torch.equal(energy, torch.ones(2, dtype=torch.float64))

    generator = torch.Generator().manual_seed(12)
    tokens = torch.randn(4, 8, 192, generator=generator)
    health = metrics.representation_health(tokens, expected_token_count=8)
    centered = tokens.double() - tokens.double().mean(dim=0, keepdim=True)
    covariance = centered.reshape(-1, 192).T.mm(centered.reshape(-1, 192)) / 31
    expected_eigenvalues = torch.linalg.eigvalsh(
        0.5 * (covariance + covariance.T)
    ).clamp_min(0.0)
    probabilities = expected_eigenvalues / expected_eigenvalues.sum()
    expected_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    assert health.effective_rank == pytest.approx(float(expected_rank))
    assert health.cross_sample_variance == pytest.approx(
        float(centered.square().sum() / (4 * 8 * 192))
    )
    collapsed = metrics.representation_health(torch.ones(2, 4, 192))
    assert collapsed.effective_rank == 0.0
    assert collapsed.cross_sample_variance == 0.0

    assert metrics.expected_training_accounting(400) == {
        "updates": 400,
        "sequence_rows": 4_000,
        "rgb_frame_presentations": 16_000,
        "online_encoder_frame_calls": 12_000,
        "ema_target_encoder_frame_calls": 4_000,
        "microbatch_graphs": 2_000,
        "backward_calls": 2_000,
        "global_gradient_clips": 400,
        "optimizer_steps": 400,
        "ema_steps": 400,
    }


def _summary(
    ratio: float,
    *,
    lower: float = 0.1,
    families: int = 8,
) -> metrics.ControlSummary:
    return metrics.ControlSummary(
        correct_macro_mean=ratio,
        control_macro_mean=1.0,
        primary_ratio=ratio,
        advantage_macro_mean=1.0 - ratio,
        advantage_bootstrap_lower_95=lower,
        positive_family_count=families,
        correct_by_scene={},
        control_by_scene={},
        advantage_by_scene={},
        advantage_by_family={},
    )


def _observation(
    update: int,
    *,
    panel_kind: str,
    panel_identity: str,
    ratio: float,
    persistence_ratio: float | None = None,
    families: int = 8,
) -> metrics.TemporalObservation:
    controls = {
        name: _summary(
            persistence_ratio if name == "persistence" and persistence_ratio else ratio,
            families=families,
        )
        for name in metrics.CONTROL_NAMES
    }
    health = metrics.RepresentationHealth(32, 256, 192, 10.0, 0.5, True)
    prediction = metrics.RepresentationHealth(32, 64, 192, 25.0, 0.5, True)
    target = metrics.RepresentationHealth(32, 64, 192, 50.0, 1.0, True)
    integrity = metrics.IntegrityFacts(
        access_and_accounting_exact=True,
        all_evaluated_finite=True,
        target_frozen_eval=True,
        target_gradient_tensor_count=0,
        ema_count=update,
        latest_training_receipt_pass=None if update == 0 else True,
        baseline_health_noncollapsed=True,
    )
    return metrics.TemporalObservation(
        update=update,
        panel_kind=panel_kind,
        panel_identity_sha256=panel_identity,
        controls=controls,
        recurrent_health=health,
        prediction_health=prediction,
        target_health=target,
        integrity=integrity,
        predecessor_controls={
            "wrong_target": _summary(0.5),
            "wrong_context": _summary(0.5),
            "position_mean": _summary(0.5),
        },
        raw_health_retentions={
            key: 0.8 for key in metrics.RAW_HEALTH_RETENTION_KEYS
        },
        place_chance_multiple_retention=0.8,
        target_place_rank_retention=0.8,
    )


def test_qualification_survival_and_continuation_are_panel_compatible() -> None:
    sentinel_id = "a" * 64
    full_id = "b" * 64
    baseline_sentinel = _observation(
        0, panel_kind="sentinel", panel_identity=sentinel_id, ratio=1.1, families=3
    )
    update_50 = _observation(
        50, panel_kind="sentinel", panel_identity=sentinel_id, ratio=1.0, families=3
    )
    decision_50 = metrics.continuation_decision(
        update_50, update_zero=baseline_sentinel
    )
    assert decision_50.passed and decision_50.action == "CONTINUE"
    invalid_full_50 = _observation(
        50, panel_kind="full", panel_identity=full_id, ratio=0.5
    )
    assert not metrics.observation_survives(invalid_full_50)
    assert (
        metrics.observation_survival_checks(invalid_full_50)[
            "registered_panel_schedule"
        ]
        is False
    )

    update_100 = _observation(
        100, panel_kind="sentinel", panel_identity=sentinel_id, ratio=0.97
    )
    decision_100 = metrics.continuation_decision(
        update_100,
        update_zero=baseline_sentinel,
        previous=update_50,
    )
    assert decision_100.passed

    baseline_full = _observation(
        0, panel_kind="full", panel_identity=full_id, ratio=1.1, families=3
    )
    passing_200 = _observation(
        200, panel_kind="full", panel_identity=full_id, ratio=0.5
    )
    assert metrics.observation_survives(passing_200)
    assert metrics.qualifies(passing_200)
    decision_200 = metrics.continuation_decision(
        passing_200, update_zero=baseline_full
    )
    assert decision_200.selected_update == 200
    assert decision_200.action == "SELECT_AND_STOP"
    assert metrics.observation_gate(passing_200)[
        "perception_temporal_qualified"
    ] is True
    assert metrics.continuation_gate(
        passing_200, update_zero=baseline_full
    )["selected_update"] == 200

    nonqualifying_200 = _observation(
        200,
        panel_kind="full",
        panel_identity=full_id,
        ratio=0.95,
        persistence_ratio=1.1,
    )
    assert not metrics.qualifies(nonqualifying_200)
    assert metrics.continuation_decision(
        nonqualifying_200, update_zero=baseline_full
    ).action == "CONTINUE"

    failed_integrity = replace(
        update_50,
        integrity=replace(
            update_50.integrity,
            access_and_accounting_exact=False,
        ),
    )
    failed = metrics.continuation_decision(
        failed_integrity, update_zero=baseline_sentinel
    )
    assert failed.action == "TERMINAL"
    assert "access_and_accounting_exact" in failed.failed_checks

    incompatible = replace(update_50, panel_identity_sha256="c" * 64)
    with pytest.raises(metrics.TemporalJepaMetricError, match="panel"):
        metrics.continuation_decision(
            incompatible, update_zero=baseline_sentinel
        )
