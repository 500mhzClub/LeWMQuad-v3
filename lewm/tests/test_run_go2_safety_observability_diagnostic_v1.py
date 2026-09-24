"""Focused source-only tests for the historical safety diagnostic runner."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.oracle import go2_scorer_failure_attribution_v1_contract as C
from lewm.oracle import go2_scorer_fit_oracle_v1_3_contract as V13C
from scripts import run_go2_safety_observability_diagnostic_v1 as R


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_closure() -> dict:
    files = {
        path: {
            "path": path,
            "sha256": _sha(path),
            "byte_count": len(path.encode("utf-8")),
        }
        for path in C.SOURCE_CLOSURE_PATHS
    }
    payload = {
        "schema": C.SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": "1" * 40,
        "source_repository_clean": True,
        "git_status_porcelain_v1": "",
        "files": files,
    }
    payload[C.SOURCE_CLOSURE_SELF_KEY] = C.canonical_digest(payload)
    return payload


def _pose(offset: float) -> dict:
    return {
        "position": [offset, 0.0, 1.0],
        "lookat": [offset + 1.0, 0.0, 1.0],
        "up": [0.0, 0.0, 1.0],
    }


def _corpus_and_prior() -> tuple[dict, dict]:
    states = []
    rows = []
    for state_index, frozen in enumerate(V13C.HISTORICAL_CALIBRATION_STATES):
        state = {**asdict(frozen), "split_role": "calibration"}
        states.append(state)
        for candidate_index in range(12):
            identity = _sha(f"{frozen.state_identity_digest}:{candidate_index}")
            rows.append({
                "split_role": "calibration",
                "branch_identity_digest": identity,
                "branch_row_digest": _sha(f"row:{identity}"),
                "assignment_identity_digest": _sha(f"assignment:{identity}"),
                **asdict(frozen),
                "candidate_index": candidate_index,
                "candidate": f"candidate-{candidate_index}",
                "primitives": ["hold"] * 4,
                "goal": {"landmark_cell": 1},
                "snapshot_digest": _sha(f"snapshot:{state_index}"),
                "realised_requested_prefix": [[[0.0, 0.0, 0.0]] * 5] * 4,
                "post_slew": [[[0.0, 0.0, 0.0]] * 5] * 4,
                "context_frames": [
                    {"camera_pose_world": _pose(float(index))}
                    for index in range(3)
                ],
                "horizon_frames": [
                    {"camera_pose_world": _pose(float(index + 3))}
                    for index in range(4)
                ],
                "proprio": [[0.0] * 30 for _ in range(15)],
                "control": [[0.0] * 2 for _ in range(15)],
                "action_context_blocks": [[0.0] * 10 for _ in range(3)],
                "previous_applied_command": [0.0, 0.0, 0.0],
                "valid": True,
                "invalid_reason": None,
                "contact_fraction": 0.0,
                "clearance_cost": 0.0,
                "stuck_fraction": 0.0,
                "fall": 0.0,
                "safety": 0.0,
                "completion": 0.0,
            })
    rows.sort(key=lambda row: row["branch_identity_digest"])
    prior = {}
    for row in rows[:12]:
        row["valid"] = False
        row["invalid_reason"] = "unlocatable_or_unreachable_geodesic"
        for field in R.AGGREGATE_LABEL_FIELDS:
            row[field] = None
        identity = row["branch_identity_digest"]
        aggregate = {field: 0.0 for field in R.AGGREGATE_LABEL_FIELDS}
        prior[identity] = {
            "path": f"prior/{identity}.json",
            "sha256": _sha(f"bytes:{identity}"),
            "replay_overlay_digest": _sha(f"overlay:{identity}"),
            "attempt_digest": _sha(f"attempt:{identity}"),
            "trace_digest": _sha(f"trace:{identity}"),
            "contact_type_evidence": "NOT_RETAINED_IN_V1_3_TRACE",
            "aggregate_label_projection": aggregate,
            "aggregate_label_projection_digest": R.digest(aggregate),
            "trace": {},
        }
    return {"state_manifest": {"states": states}, "rows": rows}, prior


def _contact_detail(category: str = "NONFOOT_GROUND") -> dict:
    return {
        "category": category,
        "robot_link_id": 13,
        "robot_link_name": "base",
        "environment_link_id": 1,
        "environment_link_name": "ground",
        "force_magnitude_n": 1.0,
    }


def _trace(*, contact_ticks: set[int] | None = None,
           clearance_ticks: set[int] | None = None,
           stuck_ticks: set[int] | None = None,
           unsafe_ticks: set[int] | None = None,
           completion_ticks: set[int] | None = None) -> dict:
    contact_ticks = contact_ticks or set()
    clearance_ticks = clearance_ticks or set()
    stuck_ticks = stuck_ticks or set()
    unsafe_ticks = unsafe_ticks or set()
    completion_ticks = completion_ticks or set()
    completion_start = min(completion_ticks) if completion_ticks else None
    assert not completion_ticks or completion_ticks == set(range(
        int(completion_start), 20))
    ticks = []
    safe = float(R.ORACLE_V12.CLEARANCE_SAFE_M)
    for tick in range(20):
        contact = tick in contact_ticks
        clearance_m = safe / 2.0 if tick in clearance_ticks else safe
        details = [_contact_detail()] if contact else []
        termination = {
            "fall": tick in unsafe_ticks,
            "out_of_bounds": False,
            "tipped": False,
            "nan": False,
        }
        ticks.append({
            "global_tick": tick,
            "block_index": tick // 5,
            "tick_in_block": tick % 5,
            "episode_id": 1,
            "episode_step": tick + 1,
            "timestamp_ns": (tick + 1) * 100_000_000,
            "requested_command": [0.0, 0.0, 0.0],
            "post_slew_command": [0.0, 0.0, 0.0],
            "position_world_xyz_m": [0.0, 0.0, 0.35],
            "quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
            "rpy_world_rad": [0.0, 0.0, 0.0],
            "xy": [0.0, 0.0],
            "yaw": 0.0,
            "z": 0.35,
            "nearest_cell_id": 1,
            "nearest_cell_distance_m": 0.0,
            "located": True,
            "accepted_cell_id": 1,
            "cell_id": 1,
            "goal_cell_id": 2,
            "raw_bfs_to_goal": 1,
            "masked_bfs_to_goal": 1,
            "geodesic_m": 1.0,
            "graph_status": "LOCATABLE_REACHABLE",
            "at_goal_cell": tick == completion_start,
            "clearance_m": clearance_m,
            "clearance_deficit": R.ORACLE_V12.clearance_deficit(clearance_m),
            "stuck": tick in stuck_ticks,
            "disallowed_contacts": len(details),
            "disallowed_contact": contact,
            "termination": termination,
            "terminated": bool(tick in unsafe_ticks),
            "nan": False,
            "completion_latched": tick in completion_ticks,
            "contact_type_evidence_status": "COMPLETE",
            "disallowed_contact_types": sorted({
                row["category"] for row in details}),
            "disallowed_contact_details": details,
        })
    start = copy.deepcopy(ticks[0])
    start.update({
        "global_tick": None, "block_index": None, "tick_in_block": None,
        "episode_step": 0, "timestamp_ns": 0, "requested_command": None,
    })
    start.pop("completion_latched")
    return {
        "schema": V13C.TRACE_SCHEMA,
        "candidate": "synthetic",
        "primitives": ["hold"] * 4,
        "requested": [[[0.0, 0.0, 0.0]] * 5] * 4,
        "post_slew": [[[0.0, 0.0, 0.0]] * 5] * 4,
        "blocks_completed": 4,
        "nan": False,
        "start": start,
        "ticks": ticks,
    }


def _identity(index: int, *, family: str = "family-a",
              stratum: str = "general") -> dict:
    return {
        "branch_identity_digest": _sha(f"branch:{index}"),
        "state_identity_digest": _sha(f"state:{index // 12}"),
        "family": family,
        "stratum": stratum,
        "candidate_index": index % 12,
    }


def test_exact_outcome_blind_plan_replays_all_288_and_only_binds_prior_12():
    corpus, prior = _corpus_and_prior()
    contract = C.build_contract(_source_closure())
    plan = R.build_plan(corpus, prior, contract)

    assert plan["state_count"] == 24
    assert plan["branch_count"] == plan["new_replay_count"] == 288
    assert plan["prior_lineage_trace_count"] == 12
    assert plan["selection"]["outcome_fields_used_for_selection"] == []
    assert plan["selection"][
        "scientific_label_fields_consulted_for_selection"] is False
    assert plan["execution"]["all_288_replayed"] is True
    assert plan["execution"]["prior_12_are_not_substituted"] is True
    assert all(row["execution_disposition"] ==
               "NEW_DIAGNOSTIC_REPLAY_REQUIRED" for row in plan["entries"])
    assert sum(row["prior_v1_3_trace"] is not None
               for row in plan["entries"]) == 12
    assert plan["replay_aggregate_equality_contract"]["source_counts"] == {
        R.V13_LABEL_SOURCE: 12, R.V2_LABEL_SOURCE: 276,
    }
    assert sum(all(value is None for value in
                   row["source_v2_aggregate_label_projection"].values())
               for row in plan["entries"]) == 12
    assert all(row["frozen_replay_target_projection_digest"]
               == R.digest(row["frozen_replay_target_projection"])
               for row in plan["entries"])
    assert plan["contract_digest"] == contract[C.CONTRACT_SELF_KEY]
    forbidden = {"label", "progress", "safety", "completion", "utility"}
    assert all(not (forbidden & set(row)) for row in plan["entries"])

    broken = copy.deepcopy(corpus)
    broken["rows"].pop()
    with pytest.raises(R.DiagnosticError, match="24x12"):
        R.build_plan(broken, prior, contract)
    duplicate = copy.deepcopy(corpus)
    duplicate["rows"][1]["branch_identity_digest"] = \
        duplicate["rows"][0]["branch_identity_digest"]
    with pytest.raises(R.DiagnosticError, match="24x12"):
        R.build_plan(duplicate, prior, contract)


def test_installed_contract_must_equal_live_clean_eight_file_closure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    closure = _source_closure()
    contract = C.build_contract(closure)
    (tmp_path / R.CONTRACT_NAME).write_text(
        json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(R, "build_source_closure", lambda **_kwargs: closure)
    assert R.load_bound_contract(out_root=tmp_path) == contract

    changed = copy.deepcopy(closure)
    changed["source_repository_commit"] = "2" * 40
    unsigned = {key: changed[key] for key in changed
                if key != C.SOURCE_CLOSURE_SELF_KEY}
    changed[C.SOURCE_CLOSURE_SELF_KEY] = C.canonical_digest(unsigned)
    monkeypatch.setattr(R, "build_source_closure", lambda **_kwargs: changed)
    with pytest.raises(R.DiagnosticError, match="current clean source"):
        R.load_bound_contract(out_root=tmp_path)


def test_contact_classifier_matches_frozen_count_and_names_all_three_types():
    contacts = {
        "link_a": np.asarray([12, 13, 12, 13, 15, 2, 13]),
        "link_b": np.asarray([1, 1, 2, 2, 16, 3, 1]),
        "force_a": np.asarray([
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [R.ORACLE_V12.CONTACT_FORCE_THRESHOLD_N / 2.0, 0.0, 0.0],
        ]),
    }
    details = R.classify_disallowed_contact_details(
        contacts, robot_link_range=(10, 20), foot_link_indices={12},
        ground_link_indices={1},
        link_names={1: "ground", 2: "wall", 12: "foot", 13: "base"},
    )
    assert [row["category"] for row in details] == list(R.CONTACT_CATEGORIES)
    assert len(details) == R.ORACLE_V12.disallowed_contact_present(
        {"link_a": contacts["link_a"], "link_b": contacts["link_b"]},
        robot_link_range=(10, 20), foot_link_indices=frozenset({12}),
        ground_link_indices=frozenset({1}),
        forces=np.linalg.norm(contacts["force_a"], axis=-1).tolist(),
    )
    assert {row["robot_link_name"] for row in details} == {"foot", "base"}
    assert R.canonical_bytes(details) == R.canonical_bytes(
        R.classify_disallowed_contact_details(
            contacts, robot_link_range=(10, 20), foot_link_indices={12},
            ground_link_indices={1},
            link_names={1: "ground", 2: "wall", 12: "foot", 13: "base"},
        ))


def test_tick_validation_prior_lineage_agreement_and_contact_limitation():
    current = _trace(contact_ticks={1, 2}, clearance_ticks={4})
    assert len(R.validate_trace_structure(
        current, require_contact_types=True)["ticks"]) == 20
    prior = copy.deepcopy(current)
    for sample in [prior["start"], *prior["ticks"]]:
        sample.pop("contact_type_evidence_status")
        sample.pop("disallowed_contact_types")
        sample.pop("disallowed_contact_details")
    agreement = R._prior_shared_field_agreement(prior, current)
    assert agreement["status"] == "PASS_SHARED_FIELDS_AGREE"
    assert agreement["contact_type_evidence_in_prior"] == \
        "NOT_RETAINED_IN_V1_3_TRACE"

    changed = copy.deepcopy(current)
    changed["ticks"][0]["position_world_xyz_m"][0] = 0.1
    with pytest.raises(R.DiagnosticError, match="tolerance"):
        R._prior_shared_field_agreement(prior, changed)
    changed_top = copy.deepcopy(current)
    changed_top["candidate"] = "different"
    with pytest.raises(R.DiagnosticError, match="top-level"):
        R._prior_shared_field_agreement(prior, changed_top)
    missing = copy.deepcopy(current)
    missing["ticks"][1]["disallowed_contact_details"] = []
    with pytest.raises(R.DiagnosticError, match="contact types"):
        R.validate_trace_structure(missing, require_contact_types=True)
    nan_tick = copy.deepcopy(current)
    nan_tick["ticks"][3]["nan"] = True
    with pytest.raises(R.DiagnosticError, match="tick 3"):
        R.validate_trace_structure(nan_tick, require_contact_types=True)
    wrong_cadence = copy.deepcopy(current)
    wrong_cadence["ticks"][3]["timestamp_ns"] += 1
    with pytest.raises(R.DiagnosticError, match="tick 3"):
        R.validate_trace_structure(wrong_cadence, require_contact_types=True)
    wrong_type = copy.deepcopy(current)
    wrong_type["ticks"][3]["global_tick"] = 3.0
    with pytest.raises(R.DiagnosticError, match="tick 3"):
        R.validate_trace_structure(wrong_type, require_contact_types=True)


def test_branch_audit_has_exact_mass_timing_runs_overlap_and_completion():
    record = R.branch_safety_observability(
        _identity(0),
        _trace(
            contact_ticks={1, 2}, clearance_ticks={2, 4}, stuck_ticks={19},
            unsafe_ticks={10}, completion_ticks=set(range(14, 20)),
        ),
    )
    assert record["safety_components"]["safety"] == 1.0
    assert record["safety_mass_attribution"]["equals_safety"] is True
    assert record["safety_mass_attribution"]["sum"] == pytest.approx(1.0)
    contact = record["component_timing"]["contact"]
    assert contact["first_tick"] == 1 and contact["last_tick"] == 2
    assert contact["contiguous_event_runs"] == [{
        "first_tick": 1, "last_tick": 2, "tick_count": 2,
        "duration_s": 0.2,
    }]
    assert contact["only_at_non_sample_ticks"] is True
    assert contact["strictly_between_sampled_horizons_only"] is False
    assert contact["pre_h1_only"] is True
    assert record["component_timing"]["clearance"][
        "sampled_endpoint_overlap"] is True
    assert record["component_timing"]["stuck"]["final_sample_active"] is True
    assert record["component_timing"]["completion"]["first_tick"] == 14
    assert record["pairwise_physical_overlap"]["contact+clearance"][
        "overlap_ticks"] == [2]
    assert record["branch_wide_safety_timing"][
        "any_evidence_at_sampled_endpoint"] is True
    assert record["branch_wide_safety_timing"]["positive_at_h4"] is True


def test_replay_aggregates_match_the_bound_frozen_scorer_target_fail_closed():
    trace = _trace(
        contact_ticks={1, 2}, clearance_ticks={4}, stuck_ticks={19},
        completion_ticks=set(range(14, 20)),
    )
    target = R.replay_aggregate_label_projection(trace)
    entry = {
        "source_v2_aggregate_label_projection": target,
        "source_v2_aggregate_label_projection_digest": R.digest(target),
        "frozen_replay_target_source_kind": R.V2_LABEL_SOURCE,
        "frozen_replay_target_projection": target,
        "frozen_replay_target_projection_digest": R.digest(target),
        "prior_v1_3_trace": None,
    }
    receipt = R.replay_aggregate_equality(entry, trace)
    assert receipt["status"] == "PASS_FROZEN_SCORER_TARGET_AGREEMENT"
    assert receipt["source_kind"] == R.V2_LABEL_SOURCE
    assert receipt["diagnostic_replay_projection"]["completion"] == 1.0
    assert receipt["absolute_tolerances"] == {
        "contact_fraction": 0.0,
        "clearance_cost": R.V13.HORIZON_POSE_ATOL,
        "stuck_fraction": 0.0,
        "fall": 0.0,
        "safety": R.V13.HORIZON_POSE_ATOL,
        "completion": 0.0,
    }

    changed = _trace(
        contact_ticks={1, 2, 3}, clearance_ticks={4}, stuck_ticks={19},
        completion_ticks=set(range(14, 20)),
    )
    with pytest.raises(R.DiagnosticError, match="contact_fraction"):
        R.replay_aggregate_equality(entry, changed)

    nulls = {field: None for field in R.AGGREGATE_LABEL_FIELDS}
    overlay_entry = {
        **entry,
        "source_v2_aggregate_label_projection": nulls,
        "source_v2_aggregate_label_projection_digest": R.digest(nulls),
        "frozen_replay_target_source_kind": R.V13_LABEL_SOURCE,
        "prior_v1_3_trace": {"aggregate_label_projection_digest": R.digest(target)},
    }
    assert R.replay_aggregate_equality(overlay_entry, trace)["source_kind"] \
        == R.V13_LABEL_SOURCE


def test_group_conditional_proportions_run_distributions_and_mass_shares():
    assert R.STRICTLY_BETWEEN_SAMPLED_HORIZON_TICKS == frozenset({
        *range(5, 9), *range(10, 14), *range(15, 19),
    })
    records = [
        R.branch_safety_observability(
            _identity(0), _trace(contact_ticks={1, 2})),
        R.branch_safety_observability(
            _identity(1), _trace(clearance_ticks={4})),
        R.branch_safety_observability(
            _identity(2), _trace(stuck_ticks={19})),
        R.branch_safety_observability(_identity(3), _trace()),
        R.branch_safety_observability(
            _identity(4), _trace(contact_ticks={5, 6})),
    ]
    summary = R._group_summary(records)
    timing = summary["safety_positive_branch_timing"]
    assert timing["safety_positive_branch_count"] == 4
    assert timing["any_evidence_at_sampled_endpoint_given_safety_positive"] \
        == pytest.approx(1 / 2)
    assert timing[
        "all_positive_evidence_only_at_non_sample_ticks_given_safety_positive"
    ] == pytest.approx(1 / 2)
    assert timing[
        "all_safety_evidence_strictly_between_sampled_horizons_given_safety_positive"
    ] == pytest.approx(1 / 4)
    assert timing["any_pre_h1_safety_evidence_given_safety_positive"] \
        == pytest.approx(1 / 4)
    assert timing["pre_h1_only_safety_evidence_given_safety_positive"] \
        == pytest.approx(1 / 4)
    assert timing["positive_at_h4_given_safety_positive"] == pytest.approx(1 / 4)
    assert timing["per_sampled_endpoint_given_safety_positive"] == {
        "4": pytest.approx(1 / 4), "9": pytest.approx(0.0),
        "14": pytest.approx(0.0), "19": pytest.approx(1 / 4),
    }
    assert summary["component_statistics"]["contact"][
        "contiguous_event_run_duration_tick_histogram"] == {"2": 2}
    shares = summary["component_share_of_summed_total_safety_mass"]
    assert sum(shares.values()) == pytest.approx(1.0)


def test_augmented_production_executor_restores_hook_without_simulation(
        monkeypatch: pytest.MonkeyPatch):
    template = _trace()
    samples = [template["start"], *template["ticks"]]
    cursor = iter(copy.deepcopy(samples))

    def original_sample(*_args, **_kwargs):
        row = next(cursor)
        row.pop("contact_type_evidence_status")
        row.pop("disallowed_contact_types")
        row.pop("disallowed_contact_details")
        return row

    def fake_executor(*_args, **_kwargs):
        return {
            **{key: copy.deepcopy(template[key]) for key in (
                "schema", "candidate", "primitives", "requested",
                "post_slew", "blocks_completed", "nan",
            )},
            "start": R.V13._trace_sample(),
            "ticks": [R.V13._trace_sample() for _ in range(20)],
        }

    robot = SimpleNamespace(links=[], get_contacts=lambda: {})
    ctx = SimpleNamespace(build=SimpleNamespace(
        robot=robot, scene=SimpleNamespace(entities=[])))
    topology = {
        "robot_link_range": (10, 20),
        "foot_link_indices": frozenset({12}),
        "ground_link_indices": frozenset({1}),
    }
    monkeypatch.setattr(R.V13, "_trace_sample", original_sample)
    monkeypatch.setattr(R.V13, "execute_branch_trace_v13", fake_executor)
    result = R.execute_diagnostic_trace(
        ctx, object(), ("synthetic", ()), field=None, topology=topology)
    assert all(row["contact_type_evidence_status"] == "COMPLETE"
               for row in result["ticks"])
    assert R.V13._trace_sample is original_sample

    def failure(*_args, **_kwargs):
        raise RuntimeError("synthetic technical failure")

    monkeypatch.setattr(R.V13, "execute_branch_trace_v13", failure)
    with pytest.raises(RuntimeError, match="technical failure"):
        R.execute_diagnostic_trace(
            ctx, object(), ("synthetic", ()), field=None, topology=topology)
    assert R.V13._trace_sample is original_sample


def test_attempt_marker_is_exclusive_and_orphan_is_terminal(tmp_path: Path):
    plan = {R.PLAN_SELF_KEY: "a" * 64, "contract_digest": "b" * 64,
            "entries": [{"branch_identity_digest": "c" * 64}]}
    marker = R.begin_attempt_once("c" * 64, plan, out_root=tmp_path)
    assert marker["maximum_attempts_for_identity"] == 1
    with pytest.raises(R.DiagnosticError, match="refusing to replace"):
        R.begin_attempt_once("c" * 64, plan, out_root=tmp_path)
    with pytest.raises(R.DiagnosticError, match="orphan.*terminal"):
        R._existing_execution_inventory(
            plan, {"rows": []}, {}, out_root=tmp_path)

    extra_root = tmp_path / "extra"
    extra_root.mkdir()
    extra_attempts = extra_root / R.ATTEMPTS_NAME
    extra_attempts.mkdir(parents=True)
    (extra_attempts / f"{'d' * 64}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(R.DiagnosticError, match="unregistered"):
        R._existing_execution_inventory(
            plan, {"rows": []}, {}, out_root=extra_root)


def test_terminal_and_audit_require_exact_288_new_rows_and_report_groups():
    identities = [_identity(index) for index in range(288)]
    inventory = [{
        **identity,
        "source_kind": "NEW_DIAGNOSTIC_REPLAY",
        "frozen_replay_target_source_kind": (
            R.V13_LABEL_SOURCE if index < 12 else R.V2_LABEL_SOURCE),
        "prior_v1_3_trace_bound": index < 12,
        "path": f"trace_rows/{identity['branch_identity_digest']}.json",
        "sha256": _sha(f"file:{index}"),
        R.TRACE_ROW_SELF_KEY: _sha(f"row:{index}"),
    } for index, identity in enumerate(identities)]
    plan = {
        R.PLAN_SELF_KEY: "a" * 64,
        "contract_digest": "b" * 64,
        "entries": [{
            **identity,
            "prior_v1_3_trace": {} if index < 12 else None,
        } for index, identity in enumerate(identities)],
    }
    terminal = R._terminal_payload(plan, inventory)
    assert terminal["new_replay_count"] == 288
    assert terminal["adopted_as_final_count"] == 0
    assert terminal["prior_lineage_trace_count"] == 12

    trace_rows = [{**identity, "trace": _trace()}
                  for identity in identities]
    audit = R.build_audit(terminal, trace_rows)
    assert audit["branch_count"] == 288
    assert audit["overall"]["branch_count"] == 288
    assert audit["frozen_scorer_target_fields_consulted"] \
        == list(R.AGGREGATE_LABEL_FIELDS)
    assert audit["scientific_corpus_labels_modified"] is False
    assert audit["tick_physical_evidence_read"] is True
    assert audit["rgb_pixels_read"] == audit["latents_accessed"] == 0
    assert "by_family" in audit and "by_stratum" in audit
    assert "by_family_stratum" in audit


def test_source_has_no_render_label_latent_or_retry_path_and_marker_is_adjacent():
    source = inspect.getsource(R)
    assert "V13.execute_branch_trace_v13" in source
    assert "V13.validate_replay_preexecution" in source
    assert "V13.validate_replay_equality" in source
    assert "TexturedV03Renderer(" not in source
    assert "score_branch(" not in source
    assert "torch.load" not in source
    assert "PIL" not in source
    start = source.index("marker = begin_attempt_once(identity, plan")
    execution = source.index("trace = execute_diagnostic_trace(", start)
    between = source[start:execution]
    assert "begin_attempt_once" in between
    assert "execute_" not in between
    assert "del ctx\n        gc.collect()" in source


def test_direct_help_works_from_arbitrary_cwd(tmp_path: Path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(Path(R.__file__).resolve()), "--help"],
        cwd=tmp_path, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "issue-plan" in result.stdout and "replay" in result.stdout
