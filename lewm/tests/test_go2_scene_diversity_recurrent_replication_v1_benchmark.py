from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as calibration
from lewm.benchmarks import go2_scene_diversity_recurrent_replication_v1 as subject
from lewm.benchmarks import go2_task_coupled_recurrent_dynamics_v1 as frozen


def _groups(role: str = "train") -> tuple[SimpleNamespace, ...]:
    rows = []
    index = 0
    for family in calibration.FAMILIES:
        for scene in range(4):
            scene_id = f"{role}-{family}-scene-{scene}"
            for state_slot in range(4):
                branches = tuple(
                    SimpleNamespace(
                        action_id=action,
                        target_rgb_artifact_id=f"{role}-target-{index}-{action}",
                        oracle_dense_rank=action,
                        labels=SimpleNamespace(
                            target_progress_m=float(8 - action),
                            fell=False,
                            tipped=False,
                        ),
                    )
                    for action in range(9)
                )
                rows.append(
                    SimpleNamespace(
                        role=role,
                        state_id=f"{role}-state-{index}",
                        family=family,
                        scene_id=scene_id,
                        group_index=index,
                        state_index_in_scene=state_slot,
                        relative_target_xy_body_m=(1.0, 0.25),
                        context_rgb_artifact_ids=tuple(
                            f"{role}-context-{index}-{frame}" for frame in range(3)
                        ),
                        branches=branches,
                    )
                )
                index += 1
    return tuple(rows)


def _role(role: str = "train") -> SimpleNamespace:
    plan = subject.build_role_feature_plan_v1(_groups(role), role=role)
    return SimpleNamespace(
        role=role,
        plan=plan,
        physical_inputs=torch.zeros(128, 9, 12, dtype=torch.float32),
        targets=torch.zeros(128, 9, 4, dtype=torch.float32),
        history_commands=torch.zeros(128, 2, 15, dtype=torch.float32),
        candidate_commands=torch.zeros(128, 9, 15, dtype=torch.float32),
        relative_goals=torch.zeros(128, 2, dtype=torch.float32),
        dense_ranks=torch.arange(9, dtype=torch.long).repeat(128, 1),
        identity_sha256=("a" if role == "train" else "b") * 64,
    )


def test_config_wraps_the_exact_frozen_protocol_and_only_changes_scene_diversity() -> None:
    config = subject.config_v1()

    assert config == {
        "schema": subject.SCHEMA,
        "frozen_recurrent_protocol": frozen.config_v1(),
        "data_intervention": {
            "scenes_per_role": 32,
            "scenes_per_family_per_role": 4,
            "states_per_scene": 4,
            "states_per_role": 128,
            "total_scenes": 64,
            "total_states": 256,
        },
    }
    assert subject.MODEL_SEEDS == frozen.MODEL_SEEDS
    assert subject.SAMPLER_SEED == frozen.SAMPLER_SEED
    assert subject.UPDATES == frozen.UPDATES == 800
    assert subject.ARM_ORDER == frozen.ARM_ORDER


def test_generalized_role_plan_retains_exact_rank_action_and_artifact_checks() -> None:
    plan = subject.build_role_feature_plan_v1(_groups(), role="train")

    assert len(plan.states) == 128
    assert len({state.scene_id for state in plan.states}) == 32
    assert len(plan.artifact_ids) == 1536
    assert tuple(plan.states[0].dense_ranks) == tuple(range(9))

    changed = list(_groups())
    branches = list(changed[0].branches)
    branches[0] = SimpleNamespace(**{**vars(branches[0]), "action_id": 8})
    changed[0] = SimpleNamespace(**{**vars(changed[0]), "branches": tuple(branches)})
    with pytest.raises(subject.SceneDiversityRecurrentReplicationError, match="nine actions"):
        subject.build_role_feature_plan_v1(changed, role="train")


def test_role_geometry_requires_four_states_in_four_scenes_per_family() -> None:
    role = _role()
    report = subject.validate_role_scene_geometry_v1(role)

    assert report["scene_count"] == 32
    assert report["family_count"] == 8
    assert report["states_per_scene"] == 4
    role.plan.states[0].scene_id if False else None  # frozen dataclass witness
    broken_states = list(role.plan.states)
    broken_states[0] = SimpleNamespace(**{
        **vars(broken_states[0]),
        "scene_id": broken_states[4].scene_id,
    })
    role.plan = SimpleNamespace(**{**vars(role.plan), "states": tuple(broken_states)})
    with pytest.raises(subject.SceneDiversityRecurrentReplicationError, match="4 states"):
        subject.validate_role_scene_geometry_v1(role)


def _comparison_rows(scene_count: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate = []
    baseline = []
    state = 0
    for family in calibration.FAMILIES:
        for scene in range(scene_count):
            for slot in range(4 if scene_count == 4 else 8):
                common = {
                    "state_id": f"state-{state}",
                    "family": family,
                    "scene_id": f"{family}-{scene}",
                }
                candidate.append({**common, "normalized_rank_regret": 0.1 + 0.001 * slot})
                baseline.append({**common, "normalized_rank_regret": 0.2})
                state += 1
    return candidate, baseline


def test_bootstrap_is_the_frozen_algorithm_and_seed_over_four_scenes() -> None:
    candidate, baseline = _comparison_rows(4)

    first = subject.paired_family_scene_bootstrap_v1(candidate, baseline)
    second = subject.paired_family_scene_bootstrap_v1(candidate, baseline)

    assert first == second
    assert first["paired_states"] == 128
    assert first["scene_clusters"] == 32
    assert first["scenes_per_family"] == {
        family: 4 for family in calibration.FAMILIES
    }
    assert first["resamples"] == subject.grounded.BOOTSTRAP_RESAMPLES
    assert first["seed"] == subject.grounded.BOOTSTRAP_SEED
    assert np.isclose(first["mean_delta"], -0.0985)

    old_candidate, old_baseline = _comparison_rows(2)
    with pytest.raises(subject.SceneDiversityRecurrentReplicationError, match="four scenes"):
        subject.paired_family_scene_bootstrap_v1(old_candidate, old_baseline)


def test_fit_delegates_without_changing_checkpoint_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role = _role()
    tokens = torch.zeros(128, 3, 256, 384, dtype=torch.float32)
    expected = {"config": frozen.config_v1(), "identity_sha256": "c" * 64}
    calls = []
    monkeypatch.setattr(
        subject.frozen,
        "fit_checkpoint_v1",
        lambda observed_role, observed_tokens, *, device: (
            calls.append((observed_role, observed_tokens, device)) or expected
        ),
    )

    observed = subject.fit_checkpoint_v1(role, tokens, device=torch.device("cpu"))

    assert observed is expected
    assert calls == [(role, tokens, torch.device("cpu"))]
