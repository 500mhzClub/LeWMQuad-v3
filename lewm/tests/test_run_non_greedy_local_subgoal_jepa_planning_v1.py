from __future__ import annotations

import math
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts import run_non_greedy_local_subgoal_jepa_planning_v1 as RUN


def _fixture(family: str, offset: int = 0):
    seed = 2_026_083_100 + RUN.FAMILIES.index(family) * 100_000 + offset
    scene = RUN.scene_spec(family, seed)
    field = RUN.geodesic_field(scene)
    rows = [RUN.simulate_candidate(scene, field, index) for index in range(12)]
    return scene, field, rows


def test_candidate_bank_and_horizon_are_exact() -> None:
    assert len(RUN.CANDIDATE_BANK) == 12
    assert [row[0] for row in RUN.CANDIDATE_BANK] == [
        "straight_fast", "straight_medium", "straight_slow", "arc_left",
        "arc_right", "turn_left", "turn_right", "turn_left_then_go",
        "turn_right_then_go", "go_then_turn_left", "reverse_then_turn", "hold",
    ]
    assert RUN.HORIZON_BLOCKS == 3
    assert RUN.TICKS_PER_BLOCK == 5
    assert RUN.COMMAND_DT_S == 0.1
    assert RUN.PHYSICS_DT_S == 0.002
    assert RUN.ROLES == {"FIT": 16, "CALIBRATION": 4, "DEVELOPMENT_HELDOUT": 4}
    assert RUN.ROLE_TOTALS == {
        "FIT": 64,
        "CALIBRATION": 16,
        "DEVELOPMENT_HELDOUT": 16,
    }


def test_each_family_fixture_is_non_greedy_eligible_without_model_data() -> None:
    for family in RUN.FAMILIES:
        scene, field, rows = _fixture(family)
        passed, evidence = RUN.eligible_state(scene, rows, field)
        assert passed, (family, evidence)
        assert evidence["geodesic_top1"] != evidence["direct_top1"]
        assert evidence["admissible_count"] >= 2
        assert evidence["path_ratio"] >= 1.25
        assert all(math.isfinite(float(row["geodesic_progress"])) for row in rows)
        assert any(
            row["oracle_viability_admissible"]
            and row["geodesic_progress"] > 0
            and row["euclidean_progress"] < 0
            for row in rows
        )


def test_nonvisual_features_are_byte_identical_across_family_and_side() -> None:
    signatures = {"base": set(), "query": set(), "anchor": set()}
    eligible_counts = {family: 0 for family in RUN.FAMILIES}
    for family_index, family in enumerate(RUN.FAMILIES):
        for offset in range(48):
            scene, field, rows = _fixture(family, offset)
            passed, _evidence = RUN.eligible_state(scene, rows, field)
            eligible_counts[family] += int(passed)
            if offset in (0, 1):
                state = {"start": scene["start"], "goal": scene["goal"]}
                base, query, anchor, _target = RUN._feature_arrays(state, rows)
                signatures["base"].add(hashlib.sha256(base.tobytes()).hexdigest())
                signatures["query"].add(hashlib.sha256(query.tobytes()).hexdigest())
                signatures["anchor"].add(hashlib.sha256(anchor.tobytes()).hexdigest())
    assert eligible_counts == {family: 48 for family in RUN.FAMILIES}
    assert {name: len(values) for name, values in signatures.items()} == {
        "base": 1,
        "query": 1,
        "anchor": 1,
    }


def test_obstacle_visible_renderer_is_deterministic_and_pose_sensitive() -> None:
    scene, _field, rows = _fixture("WALL_DETOUR")
    first = RUN.render_rgb(scene, scene["start"])
    second = RUN.render_rgb(scene, scene["start"])
    future = RUN.render_rgb(scene, rows[0]["horizon_poses"]["H3"])
    assert first.shape == (168, 224, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert not np.array_equal(first, future)
    assert len(np.unique(first.reshape(-1, 3), axis=0)) > 20


def test_occupancy_geodesic_rejects_diagonal_corner_cutting() -> None:
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[0, 1] = True
    blocked[1, 0] = True
    assert not RUN._grid_step_allowed(blocked, 0, 0, 1, 1)
    blocked[0, 1] = False
    blocked[1, 0] = False
    assert RUN._grid_step_allowed(blocked, 0, 0, 1, 1)


def test_descriptive_completion_and_backward_stuck_match_contract() -> None:
    spec = {"goal": [0.2, 0.0]}
    assert RUN._completed_at_goal(spec, [0.0, 0.0, 0.0])
    assert not RUN._completed_at_goal(spec, [-0.2, 0.2, 0.0])
    assert RUN._stuck_outcome(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [[-0.2, 0.0, 0.0]],
    )
    assert not RUN._stuck_outcome(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [[0.0, 0.0, 0.0]],
    )


def test_failed_stage_a_never_erases_conditional_stage_b(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "output"
    cache = output / "cache"
    cache.mkdir(parents=True)
    RUN.atomic_json(
        cache / "stage_a_metrics.json",
        {
            "decision": {
                "stage_a_authorizes_predictor_substitution": False,
            }
        },
    )
    conditional = output / "conditional_stage_b_scores.jsonl"
    conditional.write_bytes(b"partial\n")
    monkeypatch.setattr(RUN, "OUTPUT_ROOT", output)
    monkeypatch.setattr(RUN, "CACHE_ROOT", cache)
    monkeypatch.setattr(RUN, "PREDICTOR_ROOT", cache / "predictors")
    with pytest.raises(RUN.ExperimentError, match="conditional Stage-B evidence"):
        RUN.materialize_predictors()
    assert conditional.read_bytes() == b"partial\n"


def test_execution_contract_rejects_self_consistent_scientific_tamper() -> None:
    head = "a" * 40
    value = RUN.attach_digest(
        {
            "schema": (
                "non_greedy_local_subgoal_jepa_planning_v1.execution_contract.v1"
            ),
            "source_freeze_commit": head,
            "source_freeze_parent": RUN.PARENT_COMMIT,
            "source_freeze_subject": (
                "Freeze non-greedy local subgoal JEPA planning experiment"
            ),
            "scientific_contract": RUN.CONTRACT.build_contract(),
            "recovery_binding": {"governance_only": True},
            "output_root": str(RUN.OUTPUT_ROOT),
            "execution_runtime": RUN.require_execution_runtime(),
        }
    )
    assert RUN._validate_execution_contract(
        value,
        head=head,
        expected_recovery_binding=value["recovery_binding"],
    ) == value
    tampered = dict(value)
    tampered["scientific_contract"] = dict(value["scientific_contract"])
    tampered["scientific_contract"]["experiment_id"] = "TAMPERED"
    tampered = RUN.attach_digest(tampered)
    with pytest.raises(RUN.ExperimentError, match="current source freeze"):
        RUN._validate_execution_contract(
            tampered,
            head=head,
            expected_recovery_binding=value["recovery_binding"],
        )
    recovery_tampered = dict(value)
    recovery_tampered["recovery_binding"] = {"governance_only": False}
    recovery_tampered = RUN.attach_digest(recovery_tampered)
    with pytest.raises(RUN.ExperimentError, match="current source freeze"):
        RUN._validate_execution_contract(
            recovery_tampered,
            head=head,
            expected_recovery_binding=value["recovery_binding"],
        )


def test_feature_projection_has_registered_widths_and_no_outcome_role_input() -> None:
    scene, _field, rows = _fixture("OFFSET_PASSAGE")
    state = {"start": scene["start"], "goal": scene["goal"]}
    base, query, anchor, target = RUN._feature_arrays(state, rows)
    assert base.shape == (12, 131)
    assert query.shape == (12, 66)
    assert anchor.shape == target.shape == (12,)
    assert np.isfinite(base).all()
    assert np.isfinite(query).all()
    # The feature widths contain no predecessor 3-way route-role field.
    assert 138 not in base.shape
    assert 71 not in query.shape


def test_runner_contains_no_custom_startup_or_audit_hook() -> None:
    text = RUN.Path(RUN.__file__).read_text()
    assert "sys.addaudithook(" not in text
    assert "PREEXECUTION attempt accounting" not in text
    assert "terminal custody bundle" not in text


def test_successful_cli_exit_does_not_emit_a_traceback() -> None:
    repo_root = Path(RUN.__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    result = subprocess.run(
        [
            RUN.CONTRACT.RUNNER_INTERPRETER,
            str(Path(RUN.__file__).resolve()),
            "--help",
        ],
        cwd=repo_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert result.stderr == ""
