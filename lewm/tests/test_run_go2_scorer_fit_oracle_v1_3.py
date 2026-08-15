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
        "replay-failures", "select-calibration", "generate-calibration",
        "compose-training-view", "status",
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


def test_exact_replay_equality_checks_snapshot_actions_and_all_four_poses():
    actions = [[[0.1, 0.0, 0.0]] * 5] * 4
    old = {
        "snapshot_digest": "a" * 64,
        "realised_requested_prefix": actions,
        "post_slew": actions,
        "horizon_frames": [{"camera_pose_world": _camera_pose()} for _ in range(4)],
    }
    branch = {"requested": actions, "post_slew": actions}
    result = W.validate_replay_equality(
        old_row=old, snapshot_digest="a" * 64, branch=branch,
        horizon_camera_poses=[_camera_pose(1e-7) for _ in range(4)],
    )
    assert result["snapshot_digest_exact"] is True
    assert result["four_horizon_poses_within_tolerance"] is True
    with pytest.raises(W.WorkflowError, match="snapshot"):
        W.validate_replay_equality(
            old_row=old, snapshot_digest="b" * 64, branch=branch,
            horizon_camera_poses=[_camera_pose() for _ in range(4)],
        )
    with pytest.raises(W.WorkflowError, match="pose"):
        W.validate_replay_equality(
            old_row=old, snapshot_digest="a" * 64, branch=branch,
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
