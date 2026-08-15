"""Fail-closed, non-Genesis tests for the oracle-v1.3 data workflow."""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import pytest

from scripts import run_go2_scorer_fit_oracle_v1_3 as W


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.fixture(scope="module")
def diagnostic():
    return W.load_diagnostic()


@pytest.fixture(scope="module")
def authority(diagnostic):
    return W.build_authority({
        "source_repository_commit": "0" * 40,
        "source_repository_clean": True,
        "source_files": [],
        "source_files_digest": W.digest([]),
    }, diagnostic)


def _labels(*, final=1.5):
    progress = -1.0 if final is None else W.V12.progress_from_distances(2.0, final)
    assert progress is not None
    safety = 0.1 / 3.0
    return {
        "start_geodesic_m": 2.0,
        "final_geodesic_m": final,
        "progress": progress,
        "contact_fraction": 0.0,
        "clearance_cost": 0.1,
        "stuck_fraction": 0.0,
        "fall": 0.0,
        "safety": safety,
        "completion": 0.0,
        "utility": W.V12.composite_utility(progress, safety, 0.0),
        "min_clearance_m": 0.12,
        "evaluation_points": 20,
    }


def _v2_rows(diagnostic):
    rows = []
    for record in diagnostic["failure_inventory"]:
        rows.append({
            "branch_identity_digest": record["branch_identity_digest"],
            "branch_row_digest": record["branch_row_digest"],
            "state_id": record["state_id"],
            "state_identity_digest": record["state_identity_digest"],
            "assignment_identity_digest": record["assignment_identity_digest"],
            "candidate_index": record["candidate_index"],
            "candidate": record["candidate"],
            "primitives": record["primitives"],
            "scene_id": record["scene_id"],
            "split_role": record["split_role"],
            "valid": False,
            "invalid_reason": "unlocatable_or_unreachable_geodesic",
            "snapshot_digest": record["snapshot_digest"],
            "goal": record["designated_goal"],
            "requested": record["requested_and_post_slew_action"]["requested"],
            "post_slew": record["requested_and_post_slew_action"]["post_slew"],
            "proprio": [[0.0] * 30 for _ in range(15)],
            "control": [[0.0] * 2 for _ in range(15)],
            "action_context_blocks": [[0.0] * 10 for _ in range(3)],
            "previous_applied_command": [0.0, 0.0, 0.0],
            "context_frames": [
                {"camera_pose_world": {
                    "position": [float(index), 0.0, 0.0],
                    "lookat": [float(index) + 1.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                }} for index in range(3)
            ],
            "horizon_frames": [
                {"camera_pose_world": {
                    "position": [float(index), 0.0, 0.0],
                    "lookat": [float(index) + 1.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                }} for index in range(4)
            ],
        })
    for index in range(W.EXPECTED_V2_VALID):
        role = "fit" if index < W.EXPECTED_OLD_FIT_VALID else "calibration"
        rows.append({
            "branch_identity_digest": _sha(f"valid-branch-{index}"),
            "branch_row_digest": _sha(f"valid-row-{index}"),
            "state_id": f"valid-state-{index // 12}",
            "state_identity_digest": _sha(f"valid-state-digest-{index // 12}"),
            "assignment_identity_digest": _sha(f"valid-assignment-{index}"),
            "candidate_index": index % 12,
            "candidate": "synthetic",
            "scene_id": f"valid-scene-{index // 12}",
            "split_role": role,
            "valid": True,
            "invalid_reason": None,
            **_labels(),
        })
    return rows


def test_command_surface_is_data_only_and_has_exact_stage_counts():
    assert W.STAGES == (
        "issue-authority", "issue-replay-plan", "adopt-valid",
        "replay-failures", "issue-selector-integrity-replacement",
        "select-calibration", "generate-calibration", "compose-training-view",
        "status",
    )
    assert not any(stage.startswith(("encode", "train")) for stage in W.STAGES)
    assert not any(
        token in stage for stage in W.STAGES
        for token in ("predictor", "final-eval", "benchmark")
    )
    assert W.EXPECTED_V2_VALID == 1422
    assert W.EXPECTED_V2_INVALID == 18
    assert W.EXPECTED_FRESH_CALIBRATION_STATES == 24
    assert W.EXPECTED_FRESH_CALIBRATION_BRANCHES == 288
    assert W.EXPECTED_TRAINING_ROWS == 1440


def test_output_guard_rejects_escape_absolute_and_symlink(tmp_path: Path):
    root = tmp_path / "out"
    with pytest.raises(W.WorkflowError, match="relative descendant"):
        W.guarded_output_path("../escape.json", out_root=root)
    with pytest.raises(W.WorkflowError, match="relative descendant"):
        W.guarded_output_path(tmp_path / "absolute.json", out_root=root)
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(W.WorkflowError, match="symlink"):
        W.guarded_output_path("linked/value.json", out_root=root)


def test_output_guard_accepts_only_the_exact_registered_storage_alias(
        tmp_path: Path, monkeypatch):
    logical = tmp_path / "logical" / "managed"
    target = tmp_path / "physical" / "managed"
    logical.parent.mkdir()
    target.mkdir(parents=True)
    logical.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(W, "OUT_ROOT", logical)
    monkeypatch.setattr(
        W.V13_CONTRACT, "REGISTERED_GENERATED_TARGET_ROOT", target)
    assert W.guarded_output_path("value.json", out_root=logical) == \
        logical / "value.json"

    wrong = tmp_path / "wrong" / "managed"
    wrong.mkdir(parents=True)
    logical.unlink()
    logical.symlink_to(wrong, target_is_directory=True)
    with pytest.raises(W.WorkflowError, match="alias changed"):
        W.guarded_output_path("value.json", out_root=logical)


def test_production_output_guard_rejects_missing_or_real_root(
        tmp_path: Path, monkeypatch):
    logical = tmp_path / "logical" / "managed"
    target = tmp_path / "physical" / "managed"
    target.mkdir(parents=True)
    monkeypatch.setattr(W, "OUT_ROOT", logical)
    monkeypatch.setattr(
        W.V13_CONTRACT, "REGISTERED_GENERATED_TARGET_ROOT", target)
    with pytest.raises(W.WorkflowError, match="alias is missing"):
        W.guarded_output_path("value.json", out_root=logical)
    logical.mkdir(parents=True)
    with pytest.raises(W.WorkflowError, match="alias is missing"):
        W.guarded_output_path("value.json", out_root=logical)


def test_authority_binds_frozen_contract_and_exact_diagnostic_allowlist(
        authority, diagnostic):
    W.validate_authority(authority)
    expected = {
        row["branch_identity_digest"] for row in diagnostic["failure_inventory"]
    }
    assert expected == set(W.V13_CONTRACT.FAILED_BRANCH_ALLOWLIST)
    assert authority["exact_replay"]["identity_count"] == 18
    assert authority["exact_replay"]["identity_set_digest"] == W.digest(sorted(expected))
    assert authority["oracle_v1_3_digest"] == W._v13().oracle_digest()
    assert authority["scorer_fit_oracle_v1_3_contract_digest"] == \
        W.V13_CONTRACT.contract_digest()
    assert authority["fresh_calibration_selection"]["warmup_blocks_inclusive"] == \
        W.V13_CONTRACT.FRESH_CALIBRATION_SELECTOR["warmup_blocks_inclusive"]


def test_authority_self_digest_rejects_one_field_mutation(authority):
    changed = copy.deepcopy(authority)
    changed["training_view"]["row_count"] = 1439
    with pytest.raises(W.WorkflowError, match="self-digest"):
        W.validate_authority(changed)


def test_replay_plan_is_exact_eighteen_and_rejects_subset(
        authority, diagnostic):
    rows = _v2_rows(diagnostic)
    plan = W.build_replay_plan(authority, diagnostic, rows)
    W.validate_replay_plan(plan, authority, diagnostic, rows)
    assert len(plan["entries"]) == 18
    for entry in plan["entries"]:
        assert entry["source_context_pose_digest"] == W.digest(
            entry["source_context_camera_poses"]
        )
        assert entry["source_prebranch_witness_digest"] == W.digest(
            entry["source_prebranch_witness"]
        )
        assert entry["source_horizon_pose_digest"] == W.digest(
            entry["horizon_camera_poses"]
        )
    subset = copy.deepcopy(plan)
    subset["entries"].pop()
    subset["entry_count"] = 17
    subset.pop(W.REPLAY_PLAN_SELF_KEY)
    subset = W._with_self_digest(subset, W.REPLAY_PLAN_SELF_KEY)
    with pytest.raises(W.WorkflowError, match="replay plan changed"):
        W.validate_replay_plan(subset, authority, diagnostic, rows)


def test_replay_plan_refuses_partial_v2_inventory(authority, diagnostic):
    rows = _v2_rows(diagnostic)
    with pytest.raises(W.WorkflowError, match="1440/1422/18"):
        W.build_replay_plan(authority, diagnostic, rows[:-1])


def test_equivalence_adopts_exact_1422_with_1146_fit(
        authority, diagnostic):
    rows = _v2_rows(diagnostic)
    receipt = W.build_equivalence_receipt(authority, rows)
    W.validate_equivalence_receipt(receipt, authority, rows)
    assert receipt["compared_branch_count"] == 1422
    assert receipt["fit_branch_count"] == 1146
    assert receipt["historical_calibration_branch_count"] == 276
    assert receipt["mismatch_count"] == 0
    assert all(row["exact_equal"] is True for row in receipt["pairs"])


def test_equivalence_rejects_one_changed_legacy_label(authority, diagnostic):
    rows = _v2_rows(diagnostic)
    rows[18]["min_clearance_m"] += 0.01
    # The equality clause adopts exact V2 values; a self-consistent changed
    # source would be caught by the V2 row digest loader.  Here, demonstrate
    # that receipt comparison itself detects a post-issuance label change.
    original = _v2_rows(diagnostic)
    receipt = W.build_equivalence_receipt(authority, original)
    with pytest.raises(W.WorkflowError, match="equivalence receipt changed"):
        W.validate_equivalence_receipt(receipt, authority, rows)


def test_label_projection_allows_v13_boundary_null_but_not_missing_label():
    boundary = _labels(final=None)
    assert W._label_projection(boundary)["final_geodesic_m"] is None
    missing = dict(boundary)
    missing["utility"] = None
    with pytest.raises(W.WorkflowError, match="incomplete or non-finite"):
        W._label_projection(missing)


def _camera_pose(offset: float = 0.0):
    return {
        "position": [1.0 + offset, 2.0, 3.0],
        "lookat": [2.0 + offset, 2.0, 3.0],
        "up": [0.0, 0.0, 1.0],
    }


def test_exact_replay_uses_preserved_prebranch_witness_not_snapshot_equality():
    actions = [[[0.1, 0.0, 0.0]] * 5] * 4
    proprio = [[float(index)] * 30 for index in range(15)]
    control = [[float(index), -float(index)] for index in range(15)]
    action_context = [[float(index)] * 10 for index in range(3)]
    previous = [0.1, -0.2, 0.3]
    old = {
        "snapshot_digest": "a" * 64,
        "realised_requested_prefix": actions,
        "post_slew": actions,
        "proprio": proprio,
        "control": control,
        "action_context_blocks": action_context,
        "previous_applied_command": previous,
        "context_frames": [
            {"camera_pose_world": _camera_pose()} for _ in range(3)
        ],
        "horizon_frames": [{"camera_pose_world": _camera_pose()} for _ in range(4)],
    }
    branch = {"requested": actions, "post_slew": actions}
    preexecution = W.validate_replay_preexecution(
        old_row=old,
        replay_snapshot_digest="b" * 64,
        context_camera_poses=[_camera_pose(1e-7) for _ in range(3)],
        proprio=proprio,
        control=control,
        action_context_blocks=action_context,
        previous_applied_command=previous,
    )
    result = W.validate_replay_equality(
        old_row=old, preexecution=preexecution, branch=branch,
        horizon_camera_poses=[_camera_pose(1e-7) for _ in range(4)],
    )
    assert result["source_snapshot_digest"] == "a" * 64
    assert result["replay_snapshot_digest"] == "b" * 64
    assert result["snapshot_digest_equality_required"] is False
    assert result["proprio_history_exact"] is True
    assert result["control_history_exact"] is True
    assert result["four_horizon_poses_within_tolerance"] is True
    with pytest.raises(W.WorkflowError, match="camera pose"):
        W.validate_replay_preexecution(
            old_row=old,
            replay_snapshot_digest="b" * 64,
            context_camera_poses=[_camera_pose(1e-2) for _ in range(3)],
            proprio=proprio,
            control=control,
            action_context_blocks=action_context,
            previous_applied_command=previous,
        )
    changed_proprio = copy.deepcopy(proprio)
    changed_proprio[0][0] += 1.0
    with pytest.raises(W.WorkflowError, match="proprio"):
        W.validate_replay_preexecution(
            old_row=old,
            replay_snapshot_digest="b" * 64,
            context_camera_poses=[_camera_pose() for _ in range(3)],
            proprio=changed_proprio,
            control=control,
            action_context_blocks=action_context,
            previous_applied_command=previous,
        )
    with pytest.raises(W.WorkflowError, match="camera pose"):
        W.validate_replay_equality(
            old_row=old, preexecution=preexecution, branch=branch,
            horizon_camera_poses=[_camera_pose(1e-2) for _ in range(4)],
        )


def test_attempt_marker_is_at_most_once_and_never_targets_v2(tmp_path: Path):
    root = tmp_path / "oracle_v1_3"
    identity = "a" * 64
    marker = W.begin_attempt_once("replay", identity, {"x": 1}, out_root=root)
    assert marker["status"] == "ATTEMPT_STARTED_NO_RETRY_AUTHORITY"
    assert (root / "replay_attempts" / f"{identity}.json").is_file()
    with pytest.raises(W.WorkflowError, match="retry/replacement refused"):
        W.begin_attempt_once("replay", identity, {"x": 1}, out_root=root)
    assert root not in W.V2_ROOT.parents and W.V2_ROOT != root
    assert W._replay_overlay_path(identity, out_root=root).parent == root / "replay_overlays"


def _old_state_manifest():
    return {
        "state_manifest_digest": "d" * 64,
        "states": [
            {"state_id": f"old-{index}", "state_identity_digest": _sha(f"old-{index}"),
             "scene_id": f"old-scene-{index}",
             "split_role": "fit" if index < 96 else "calibration"}
            for index in range(120)
        ],
    }


def _fresh_states():
    return [
        {"state_id": f"fresh-{family}-{stratum}", "family": family,
         "stratum": stratum, "scene_id": f"fresh-scene-{family}-{stratum}"}
        for family in W.FAMILIES for stratum in W.STRATA
    ]


def _selection_fixture(tmp_path: Path, authority):
    pool = {}
    for family in W.FAMILIES:
        scenes = []
        for index in range(3):
            scene = tmp_path / "corpus" / "development" / family / (
                f"fresh-{family}-{index}"
            )
            scene.mkdir(parents=True)
            (scene / "manifest.json").write_text("{}\n")
            scenes.append(scene)
        pool[family] = scenes
    correction = {
        W.V13_CONTRACT.SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY: "c" * 64,
    }
    old = _old_state_manifest()
    plan = {W.REPLAY_PLAN_SELF_KEY: "p" * 64}
    equivalence = {W.EQUIVALENCE_SELF_KEY: "e" * 64}
    overlays = {W.REPLAY_OVERLAY_MANIFEST_SELF_KEY: "o" * 64}
    attempt = W.build_fresh_selection_attempt(
        authority=authority, correction=correction, v2_state_manifest=old,
        plan=plan, equivalence=equivalence, overlay_manifest=overlays,
        pool=pool, exclusion={"synthetic": True},
    )
    W.validate_fresh_selection_attempt(
        attempt, authority=authority, correction=correction,
        v2_state_manifest=old, plan=plan, equivalence=equivalence,
        overlay_manifest=overlays, pool=pool,
        exclusion={"synthetic": True},
    )
    return correction, attempt, pool


def _selected_for_task(task):
    return {
        "state_id": task["state_id"],
        "family": task["family"],
        "scene_id": task["scene_id"],
        "scene_dir": task["scene_dir"],
        "scene_manifest_sha256": task["scene_manifest_sha256"],
        "scene_manifest_byte_count": task["scene_manifest_byte_count"],
        "split": "development",
        "drive_seed": task["drive_seed"],
        "stratum": task["stratum"],
        "warmup_blocks": 40,
        "source_step": 200,
        "episode_id": 1,
        "episode_cluster_id": f"{task['scene_id']}/env0/ep1",
        "cell_id": 2,
        "boundary": {"source_step": 200},
        "goal": {"material_id": "goal"},
        "goal_type": "goal",
        "body_clearance_m": 0.3,
        "clearance_m": 0.2,
        "previous_applied_command": [0.0, 0.0, 0.0],
    }


def _complete_selection_sequence(tmp_path: Path, authority):
    correction, attempt, pool = _selection_fixture(tmp_path, authority)
    tasks, results = [], []
    used = set()
    for slot, (family, stratum) in enumerate(
        (family, stratum)
        for family in W.FAMILIES for stratum in W.STRATA
    ):
        scene = next(path for path in pool[family] if path.name not in used)
        task = W.build_fresh_selection_task(
            attempt=attempt, correction=correction, family=family,
            stratum=stratum, slot_index=slot, scene_ordinal=0,
            scene_dir=scene, used_scene_ids=sorted(used),
        )
        result = W.build_fresh_selection_result(
            task=task, selected_state=_selected_for_task(task),
        )
        tasks.append(task)
        results.append(result)
        used.add(scene.name)
    return correction, attempt, tasks, results


def test_fresh_manifest_is_exact_disjoint_8x3_and_frozen_before_branches(authority):
    manifest = W.build_fresh_calibration_manifest(
        authority=authority, v2_state_manifest=_old_state_manifest(),
        states=_fresh_states(), exclusion_binding={"synthetic": True},
    )
    W.validate_fresh_calibration_manifest(
        manifest, authority=authority, v2_state_manifest=_old_state_manifest()
    )
    assert manifest["state_count"] == 24
    assert manifest["candidate_count"] == 288
    assert manifest["candidate_outcomes_consumed"] is False
    assert manifest["all_identities_frozen_before_branch_execution"] is True
    assert {(row["family"], row["stratum"]) for row in manifest["states"]} == {
        (family, stratum) for family in W.FAMILIES for stratum in W.STRATA
    }


def test_fresh_manifest_refuses_old_scene_or_duplicate_slot(authority):
    states = _fresh_states()
    states[0]["scene_id"] = "old-scene-0"
    with pytest.raises(W.WorkflowError, match="disjoint 8x3"):
        W.build_fresh_calibration_manifest(
            authority=authority, v2_state_manifest=_old_state_manifest(),
            states=states, exclusion_binding={},
        )
    states = _fresh_states()
    states[-1]["stratum"] = states[-2]["stratum"]
    with pytest.raises(W.WorkflowError, match="disjoint 8x3"):
        W.build_fresh_calibration_manifest(
            authority=authority, v2_state_manifest=_old_state_manifest(),
            states=states, exclusion_binding={},
        )


def test_selector_integrity_sequence_preserves_family_stratum_lexical_order(
        tmp_path: Path, authority):
    correction, attempt, tasks, results = _complete_selection_sequence(
        tmp_path, authority
    )
    states = W._validate_fresh_selection_sequence(
        attempt=attempt, correction=correction, tasks=tasks, results=results,
    )
    assert len(states) == 24
    assert len({row["scene_id"] for row in states}) == 24
    assert [(row["family"], row["stratum"]) for row in states] == [
        (family, stratum)
        for family in W.FAMILIES for stratum in W.STRATA
    ]

    skipped = copy.deepcopy(tasks)
    skipped[0]["scene_ordinal"] = 1
    skipped[0].pop(W.FRESH_SELECTION_TASK_SELF_KEY)
    skipped[0] = W._with_self_digest(
        skipped[0], W.FRESH_SELECTION_TASK_SELF_KEY
    )
    changed_result = W.build_fresh_selection_result(
        task=skipped[0], selected_state=_selected_for_task(skipped[0]),
    )
    with pytest.raises(W.WorkflowError, match="lexical sequence"):
        W._validate_fresh_selection_sequence(
            attempt=attempt, correction=correction,
            tasks=[skipped[0], *tasks[1:]],
            results=[changed_result, *results[1:]],
        )


def test_selector_result_surface_is_exact_and_candidate_outcomes_are_forbidden(
        tmp_path: Path, authority):
    correction, attempt, tasks, results = _complete_selection_sequence(
        tmp_path, authority
    )
    W.validate_fresh_selection_task(
        tasks[0], attempt=attempt, correction=correction,
    )
    W.validate_fresh_selection_result(results[0], task=tasks[0])
    changed = copy.deepcopy(results[0])
    changed["selected_state"]["candidate_outcome"] = 1.0
    changed.pop(W.FRESH_SELECTION_RESULT_SELF_KEY)
    changed = W._with_self_digest(changed, W.FRESH_SELECTION_RESULT_SELF_KEY)
    with pytest.raises(W.WorkflowError, match="selected state changed"):
        W.validate_fresh_selection_result(changed, task=tasks[0])
    changed = copy.deepcopy(results[0])
    changed["extra"] = True
    changed.pop(W.FRESH_SELECTION_RESULT_SELF_KEY)
    changed = W._with_self_digest(changed, W.FRESH_SELECTION_RESULT_SELF_KEY)
    with pytest.raises(W.WorkflowError, match="scene result changed"):
        W.validate_fresh_selection_result(changed, task=tasks[0])
    changed_task = copy.deepcopy(tasks[0])
    changed_task["extra"] = True
    changed_task.pop(W.FRESH_SELECTION_TASK_SELF_KEY)
    changed_task = W._with_self_digest(
        changed_task, W.FRESH_SELECTION_TASK_SELF_KEY,
    )
    with pytest.raises(W.WorkflowError, match="scene task changed"):
        W.validate_fresh_selection_task(
            changed_task, attempt=attempt, correction=correction,
        )


def test_selector_success_terminal_binds_every_durable_task_and_result(
        tmp_path: Path, authority):
    correction, attempt, tasks, results = _complete_selection_sequence(
        tmp_path, authority
    )
    out_root = tmp_path / "out"
    for task, result in zip(tasks, results):
        task_path, created = W._publish_exact_selection_task(
            task, out_root=out_root
        )
        launch_path, launch = W._claim_fresh_selection_launch(
            task=task, attempt=attempt, correction=correction,
            out_root=out_root,
        )
        result_path = W._selection_result_path(
            task[W.FRESH_SELECTION_TASK_SELF_KEY], out_root=out_root,
        )
        W._atomic_json(result_path, result, out_root=out_root)
        assert created is True
        assert launch[W.FRESH_SELECTION_LAUNCH_SELF_KEY]
        assert task_path.is_file() and launch_path.is_file() and result_path.is_file()
    terminal = W.build_fresh_selection_terminal(
        attempt=attempt, correction=correction, tasks=tasks, results=results,
    )
    states = W.validate_fresh_selection_terminal(
        terminal, attempt=attempt, correction=correction, out_root=out_root,
    )
    assert terminal["task_result_count"] == 24
    assert terminal["selected_state_count"] == 24
    assert len(states) == 24


def test_selector_unresolved_child_terminal_is_permanent(
        tmp_path: Path, authority):
    correction, attempt, tasks, _results = _complete_selection_sequence(
        tmp_path, authority
    )
    terminal = W.build_failed_fresh_selection_terminal(
        attempt=attempt, correction=correction,
        failed_task=tasks[0], return_code=-11,
    )
    out_root = tmp_path / "out"
    W._publish_exact_selection_task(tasks[0], out_root=out_root)
    with pytest.raises(W.WorkflowError, match="terminal after unresolved child"):
        W.validate_fresh_selection_terminal(
            terminal, attempt=attempt, correction=correction,
            out_root=out_root,
        )
    assert terminal["retry_or_scene_replacement_authorised"] is False
    assert terminal["candidate_branch_execution_started"] is False


def test_selector_restart_never_relaunches_preexisting_task_without_result(
        tmp_path: Path, authority):
    _correction, _attempt, tasks, _results = _complete_selection_sequence(
        tmp_path, authority
    )
    out_root = tmp_path / "out"
    _path, first_created = W._publish_exact_selection_task(
        tasks[0], out_root=out_root
    )
    _path, second_created = W._publish_exact_selection_task(
        tasks[0], out_root=out_root
    )
    assert first_created is True
    assert second_created is False
    source = inspect.getsource(W.select_fresh_calibration_states)
    assert source.index("if not task_created and not result_path.exists()") \
        < source.index("_run_fresh_selection_child")


def test_selector_launch_claim_is_exclusive(
        tmp_path: Path, authority):
    correction, attempt, tasks, _results = _complete_selection_sequence(
        tmp_path, authority
    )
    out_root = tmp_path / "out"
    first_path, first = W._claim_fresh_selection_launch(
        task=tasks[0], attempt=attempt, correction=correction,
        out_root=out_root,
    )
    W.validate_fresh_selection_launch_marker(
        first, task=tasks[0], attempt=attempt, correction=correction,
    )
    assert first_path.is_file()
    with pytest.raises(W.WorkflowError, match="refusing to replace"):
        W._claim_fresh_selection_launch(
            task=tasks[0], attempt=attempt, correction=correction,
            out_root=out_root,
        )


def test_selector_worker_refuses_to_rerun_an_existing_result(tmp_path: Path):
    out_root = tmp_path / "out"
    tasks = out_root / "fresh_calibration/selection_tasks"
    results = out_root / "fresh_calibration/selection_results"
    tasks.mkdir(parents=True)
    results.mkdir(parents=True)
    identity = "a" * 64
    task_path = tasks / f"{identity}.json"
    result_path = results / f"{identity}.json"
    task_path.write_text("{}\n")
    result_path.write_text("{}\n")
    with pytest.raises(W.WorkflowError, match="existing result"):
        W.run_fresh_selection_worker(
            task_path=task_path, result_path=result_path, backend="cpu",
        )


def test_selector_worker_refuses_duplicate_manual_launch_marker(tmp_path: Path):
    out_root = tmp_path / "out"
    tasks = out_root / "fresh_calibration/selection_tasks"
    results = out_root / "fresh_calibration/selection_results"
    tasks.mkdir(parents=True)
    results.mkdir(parents=True)
    identity = "b" * 64
    task_path = tasks / f"{identity}.json"
    result_path = results / f"{identity}.json"
    task_path.write_text("{}\n")
    W._selection_launch_path(identity, out_root=out_root).write_text("{}\n")
    with pytest.raises(W.WorkflowError, match="existing launch marker"):
        W.run_fresh_selection_worker(
            task_path=task_path, result_path=result_path, backend="cpu",
        )


def test_selector_worker_refuses_any_existing_terminal(tmp_path: Path):
    out_root = tmp_path / "out"
    tasks = out_root / "fresh_calibration/selection_tasks"
    results = out_root / "fresh_calibration/selection_results"
    tasks.mkdir(parents=True)
    results.mkdir(parents=True)
    identity = "c" * 64
    task_path = tasks / f"{identity}.json"
    result_path = results / f"{identity}.json"
    task_path.write_text("{}\n")
    (out_root / "fresh_calibration/selection_terminal.json").write_text("{}\n")
    with pytest.raises(W.WorkflowError, match="existing terminal"):
        W.run_fresh_selection_worker(
            task_path=task_path, result_path=result_path, backend="cpu",
        )


def test_fresh_branch_generation_requires_all_three_continuation_gates():
    loader = inspect.getsource(W.load_validated_fresh_selection)
    assert "validate_fresh_selection_terminal" in loader
    assert "validate_fresh_calibration_manifest" in loader
    assert "load_selector_integrity_replacement_authority" in loader
    assert "continuation_requires_current_successor_preregistration_binding" \
        in loader
    generation = inspect.getsource(W.generate_fresh_calibration)
    assert "load_validated_fresh_selection" in generation
    assert generation.index("load_validated_fresh_selection") \
        < generation.index("begin_attempt_once")


def test_historical_calibration_disposition_is_preserved_not_discarded():
    states = _old_state_manifest()["states"]
    rows = [
        {"split_role": "calibration" if index < 288 else "fit"}
        for index in range(1440)
    ]
    disposition = W._historical_calibration_disposition(states, rows)
    assert disposition["state_count"] == 24
    assert disposition["branch_count"] == 288
    assert disposition["status"] == "DEVELOPMENT_ONLY"
    assert disposition["qualification_eligible"] is False
    assert disposition["discarded"] is False


def test_training_view_shape_binds_contract_and_zero_missing_labels():
    disposition = W._with_self_digest({
        "state_count": 24, "branch_count": 288,
        "status": "DEVELOPMENT_ONLY", "qualification_eligible": False,
        "discarded": False, "state_identity_digests": [_sha("state")],
        "scene_ids": ["scene"],
    }, "disposition_digest")
    rows = []
    for index in range(1152):
        source = (W.SOURCE_KIND_V2_VALID if index < 1146
                  else W.SOURCE_KIND_REPLAY)
        rows.append(W._with_self_digest({
            "role": "fit", "source_kind": source,
            "state_identity_digest": _sha(f"fit-state-{index // 12}"),
            "scene_id": f"fit-scene-{index // 12}",
            "branch_identity_digest": _sha(f"fit-branch-{index}"),
            "label_projection": _labels(),
        }, "training_view_row_digest"))
    for index in range(288):
        rows.append(W._with_self_digest({
            "role": "calibration", "source_kind": W.SOURCE_KIND_FRESH,
            "state_identity_digest": _sha(f"fresh-state-{index // 12}"),
            "scene_id": f"fresh-scene-{index // 12}",
            "branch_identity_digest": _sha(f"fresh-branch-{index}"),
            "label_projection": _labels(),
        }, "training_view_row_digest"))
    view = W._with_self_digest({
        "schema": W.TRAINING_VIEW_SCHEMA, "status": W.STATUS, "complete": True,
        "scorer_fit_oracle_v1_3_contract_digest": W.V13_CONTRACT.contract_digest(),
        "fit_state_count": 96, "fit_branch_count": 1152,
        "calibration_state_count": 24, "calibration_branch_count": 288,
        "row_count": 1440, "missing_label_count": 0,
        "historical_calibration_disposition": disposition,
        "rows": rows,
    }, W.TRAINING_VIEW_SELF_KEY)
    W._validate_training_view_shape(view)
    changed = copy.deepcopy(view)
    changed["missing_label_count"] = 1
    changed.pop(W.TRAINING_VIEW_SELF_KEY)
    changed = W._with_self_digest(changed, W.TRAINING_VIEW_SELF_KEY)
    with pytest.raises(W.WorkflowError, match="shape/count"):
        W._validate_training_view_shape(changed)


def test_simulator_stage_requires_explicit_authorized_flag(monkeypatch):
    reached = []
    monkeypatch.setattr(W, "generate_replay_overlays", lambda **_kwargs: reached.append(True))
    with pytest.raises(W.WorkflowError, match="requires --execute-authorized"):
        W.main(["--stage", "replay-failures"])
    assert reached == []


def test_status_explicitly_exposes_no_encoder_predictor_or_final_benchmark(tmp_path):
    value = W.status(out_root=tmp_path / "absent")
    assert value["runtime_running"] is False
    assert value["encoder_or_trainer_exposed"] is False
    assert value["predictor_or_final_benchmark_exposed"] is False
    assert not any(value["artifacts"].values())


def test_fresh_selector_uses_scorer_fit_pool_without_final_eval_authority():
    source = inspect.getsource(W.select_fresh_calibration_states)
    assert 'B.scene_pool("scorer_fit")' in source
    assert 'B.scene_pool("final_eval")' not in source


def test_replay_uses_the_verified_raw_state_manifest_without_runtime_authority():
    source = inspect.getsource(W.generate_replay_overlays)
    assert 'corpus["state_manifest"]["states"]' in source
    assert "load_full_bank_v2_branch_runtime_authority" not in source
    assert "_load_runtime_v2_authority" not in source


def test_every_physical_stage_checks_the_frozen_genesis_runtime():
    for function in (
        W.generate_replay_overlays,
        W.select_fresh_calibration_states,
        W.generate_fresh_calibration,
    ):
        assert "require_genesis_runtime()" in inspect.getsource(function)


def test_overlay_validator_recomputes_physical_witness_not_only_flags():
    source = inspect.getsource(W._validate_replay_overlay_binding)
    assert "_normalised_prebranch_witness(**source_witness)" in source
    assert "_normalised_prebranch_witness(**replay_witness)" in source
    assert "_camera_pose_sequence_error" in source
    assert "physical_witness_matches" in source
    assert 'trace.get("requested") != planned.get("requested")' in source
