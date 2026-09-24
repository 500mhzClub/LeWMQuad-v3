"""Outcome-free tests for the scorer-fit state-selector successor."""
from __future__ import annotations

import math
from types import SimpleNamespace
import weakref

import pytest

from scripts import build_go2_branch_corpus_v1_2 as B


@pytest.fixture(autouse=True)
def _legacy_successor_selection_binding(monkeypatch):
    """Keep synthetic V1 census shards on their explicit test binding.

    The production builder now rejects any legacy shard that is not also bound
    to the live V2 selection.  These tests exercise the frozen V1 reducer with
    the synthetic ``b*64`` binding supplied throughout this file.
    """

    monkeypatch.setattr(B, "selection_digest", lambda: "b" * 64)


def _status(**overrides):
    status = {
        "task_completed": False,
        "goal_claimed": False,
        "terminated": False,
        "truncated": False,
        "termination_flags": {
            "fall": False, "out_of_bounds": False, "tipped": False,
            "nan": False,
        },
    }
    status.update(overrides)
    return status


def _eligibility(*, hops=0, distance=0.5, bearing=0.0, status=None):
    return B._predecessor_start_radius_eligibility(
        graph_hops=hops,
        reachable=True,
        continuous_geodesic_m=distance,
        bearing_body_rad=bearing,
        task_status=_status() if status is None else status,
    )


def _scene_row(family: str, scene_index: int, *, hops: int = 0,
               stratum: str = "completion_enriched"):
    return {
        "family": family,
        "scene_id": f"{family}-{scene_index:03d}",
        "stratum": stratum,
        "first_eligible_block": 40 + scene_index,
        "continuous_geodesic_m": 0.5,
        "abs_bearing_rad": 0.25,
        "graph_hops_diagnostic": hops,
        "body_clearance_m": 0.2,
        # This value is a tripwire: the reducer is deliberately unable to
        # consume candidate or branch outcomes.
        "branch_outcome": object(),
    }


def _source():
    return {
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "bound_implementations_digest": "c" * 64,
    }


def _task(family: str, scene_index: int):
    payload = {
        "schema": "go2_scorer_fit_selector_feasibility_scene_task_v1",
        "family": family,
        "task_index_within_family": scene_index,
        "scene_id": f"{family}-{scene_index:03d}",
        "scene_dir": f"/synthetic/{family}/{scene_index:03d}",
        "split": "synthetic",
        "drive_seed": scene_index,
        "scene_manifest_sha256": f"{scene_index:064x}",
        "scene_manifest_byte_count": 100 + scene_index,
        "genesis_scene_sha256": f"{scene_index + 1000:064x}",
        "genesis_scene_byte_count": 200 + scene_index,
        "requested_strata": list(B.STRATA),
    }
    payload["scene_task_digest"] = B.canonical_digest(payload)
    return payload


def _task_census(*, scene_count: int = 5,
                 exclusion_digest: str = "d" * 64):
    families = []
    all_digests = []
    for family in B.STATE_SELECTOR.REQUIRED_FAMILIES:
        tasks = [_task(family, index) for index in range(scene_count)]
        digests = [task["scene_task_digest"] for task in tasks]
        all_digests.extend(digests)
        families.append({
            "family": family,
            "allowed_scene_count": len(tasks),
            "tasks": tasks,
            "family_task_set_digest": B.canonical_digest(digests),
        })
    payload = {
        "schema": B.SELECTOR_FEASIBILITY_TASK_CENSUS_SCHEMA,
        "status": "FROZEN_OUTCOME_FREE_EXHAUSTIVE_SCENE_TASK_CENSUS",
        "complete": True,
        "source_repository_commit": _source()["source_repository_commit"],
        "clean_source_binding_digest": B.canonical_digest(_source()),
        "bound_implementations_digest":
            _source()["bound_implementations_digest"],
        "successor_selection_digest": "b" * 64,
        "state_selector_amendment_digest":
            B.STATE_SELECTOR.state_selector_amendment_digest(),
        "exclusion_binding_digest": exclusion_digest,
        "family_count": len(families),
        "scene_task_count": len(all_digests),
        "families": families,
        "scene_task_set_digest": B.canonical_digest(all_digests),
        "selected_state_identities_created": False,
        "candidate_outcomes_loaded": False,
        "branch_identities_created": False,
        "branches_attempted": 0,
        "frames_rendered": 0,
        "target_latents_encoded": 0,
        "scorer_training_started": False,
    }
    payload["state_selector_feasibility_task_census_digest"] = \
        B.canonical_digest(payload)
    return payload


def _scene_shard(task, census, *, include_completion=True, runtime_s=0.2,
                 exclusion_digest: str = "d" * 64):
    evidence = []
    for stratum in B.STRATA:
        if stratum == "completion_enriched" and not include_completion:
            continue
        row = _scene_row(
            task["family"], int(task["task_index_within_family"]),
            stratum=stratum)
        row.pop("branch_outcome")
        evidence.append(row)
    result = {
        "family": task["family"],
        "scene_id": task["scene_id"],
        "scene_evidence": evidence,
        "rejection_counts": {"snapshot_unavailable": 1},
    }
    return B._build_selector_feasibility_scene_shard(
        task=task, scene_result=result,
        task_census_digest=census[
            "state_selector_feasibility_task_census_digest"],
        source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest=exclusion_digest, runtime_s=runtime_s)


def _family_reduction(family: str, *, passed: bool = True,
                      runtime_s: float = 1.0,
                      exclusion_digest: str = "d" * 64, census=None,
                      out=None):
    census = (_task_census(exclusion_digest=exclusion_digest)
              if census is None else census)
    tasks = B._selector_feasibility_family_tasks(census, family)
    shards = [
        _scene_shard(
            task, census,
            include_completion=passed or index < len(tasks) - 1,
            runtime_s=runtime_s / len(tasks),
            exclusion_digest=exclusion_digest)
        for index, task in enumerate(tasks)
    ]
    if out is not None:
        for task, shard in zip(tasks, shards):
            B.atomic_json(
                B._selector_feasibility_scene_shard_path(out, task), shard)
    return B._reduce_selector_feasibility_family_scene_shards(
        family=family, tasks=tasks, shards=shards,
        task_census_digest=census[
            "state_selector_feasibility_task_census_digest"],
        source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest=exclusion_digest)


def test_hops_zero_not_completed_meeting_continuous_contract_is_eligible():
    result = _eligibility(hops=0, distance=0.5, bearing=0.0)
    assert result["eligible"] is True
    assert result["graph_hops_diagnostic"] == 0
    assert result["rejection_reasons"] == []


@pytest.mark.parametrize(
    ("flag", "reason"),
    (
        ("task_completed", "completion_snapshot_task_completed"),
        ("goal_claimed", "completion_snapshot_goal_claimed"),
        ("terminated", "completion_snapshot_terminated"),
        ("truncated", "completion_snapshot_truncated"),
    ),
)
def test_completed_claimed_terminated_or_truncated_snapshot_is_rejected(flag, reason):
    result = _eligibility(status=_status(**{flag: True}))
    assert result["eligible"] is False
    assert reason in result["rejection_reasons"]


def test_missing_snapshot_task_status_is_rejected_fail_closed():
    result = _eligibility(status={})
    assert result["eligible"] is False
    assert set(result["rejection_reasons"]) == {
        "completion_snapshot_task_completed_unavailable",
        "completion_snapshot_goal_claimed_unavailable",
        "completion_snapshot_terminated_unavailable",
        "completion_snapshot_truncated_unavailable",
    }


def test_snapshot_task_status_matches_production_route_completion(monkeypatch):
    class Policy:
        revisit_after_arrival = False

        @staticmethod
        def visited_landmark_cells(env_idx):
            assert env_idx == 0
            return frozenset({3, 5})

    runner = SimpleNamespace(
        _scheduler=SimpleNamespace(policy_for=lambda env_idx: Policy()),
        _blocks_in_episode=[40],
        _scene_graph=object(),
        _landmark_cell_to_id={3: "a", 5: "b"},
    )
    ctx = SimpleNamespace(runner=runner)
    monkeypatch.setattr(
        B.V1, "_termination_flags",
        lambda _ctx: {"fall": False, "nan": False})
    status = B._snapshot_task_status(ctx, 5)
    assert status["goal_claimed"] is True
    assert status["task_completed"] is True
    assert status["terminated"] is False
    assert status["truncated"] is False
    assert B._snapshot_claim_semantics_unchanged(status) is True
    assert B._production_task_reset_semantics_unchanged(status) is True

    changed_claim = dict(status)
    changed_claim["goal_claimed"] = False
    assert B._snapshot_claim_semantics_unchanged(changed_claim) is False
    changed_reset = dict(status)
    changed_reset["task_completed"] = False
    assert B._production_task_reset_semantics_unchanged(changed_reset) is False


@pytest.mark.parametrize(
    ("blocks", "revisit", "visited", "expected_completed"),
    (
        (0, False, {3, 5}, False),
        (40, True, {3, 5}, False),
        (40, False, {3}, False),
        (40, False, {3, 5}, True),
    ),
)
def test_task_reset_check_is_distinct_from_designated_goal_claim(
        monkeypatch, blocks, revisit, visited, expected_completed):
    class Policy:
        revisit_after_arrival = revisit

        @staticmethod
        def visited_landmark_cells(_env_idx):
            return frozenset(visited)

    runner = SimpleNamespace(
        _scheduler=SimpleNamespace(policy_for=lambda _env_idx: Policy()),
        _blocks_in_episode=[blocks],
        _scene_graph=object(),
        _landmark_cell_to_id={3: "a", 5: "b"},
    )
    ctx = SimpleNamespace(runner=runner)
    monkeypatch.setattr(B.V1, "_termination_flags", lambda _ctx: {"nan": False})
    status = B._snapshot_task_status(ctx, 3)
    assert status["goal_claimed"] is (3 in visited)
    assert status["task_completed"] is expected_completed
    assert B._snapshot_claim_semantics_unchanged(status) is True
    assert B._production_task_reset_semantics_unchanged(status) is True


def test_oracle_completion_target_binding_is_not_a_snapshot_task_flag():
    assert B._oracle_completion_target_unchanged() is True
    assert B.v12_oracle_digest() == B.STATE_SELECTOR.ORACLE_V1_2_DIGEST


@pytest.mark.parametrize(
    ("hops", "distance", "bearing", "reason"),
    (
        (0, 0.750001, 0.0, "completion_geodesic_gt_0_75m"),
        (4, 0.750001, 0.0, "completion_geodesic_gt_0_75m"),
        (0, 0.5, math.radians(75.0) + 1e-9, "completion_bearing_gt_75deg"),
        (4, 0.5, math.radians(75.0) + 1e-9, "completion_bearing_gt_75deg"),
    ),
)
def test_unchanged_continuous_conditions_reject_regardless_of_hops(
        hops, distance, bearing, reason):
    result = _eligibility(hops=hops, distance=distance, bearing=bearing)
    assert result["eligible"] is False
    assert reason in result["rejection_reasons"]


def test_valid_positive_hops_completion_remains_eligible():
    result = _eligibility(hops=3, distance=0.6, bearing=math.radians(30.0))
    assert result["eligible"] is True
    assert result["graph_hops_diagnostic"] == 3


def test_exact_0_75_threshold_is_preserved_without_tolerance():
    assert B.COMPLETION_ENRICHED_MAX_GEODESIC_M == 0.75
    assert _eligibility(distance=0.75)["eligible"] is True
    assert _eligibility(distance=math.nextafter(0.75, math.inf))["eligible"] is False


def test_eligibility_and_dry_run_reducer_have_no_branch_outcome_read(monkeypatch):
    monkeypatch.setattr(
        B, "_outcome_generation_started",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run touched outcome plumbing")))
    result = _eligibility()
    summary = B.build_selector_feasibility_summary(
        family="rough_local_dynamics", allowed_scene_count=1,
        requested_strata=("completion_enriched",),
        scene_evidence=[_scene_row("rough_local_dynamics", 0)],
        rejection_counts={})
    assert result["eligible"] is True
    assert summary["per_stratum"]["completion_enriched"][
        "eligible_distinct_scenes"] == 1


def test_dry_run_can_prove_five_completion_scenes_for_rough_and_open():
    for family in ("rough_local_dynamics", "open_obstacle_field"):
        summary = B.build_selector_feasibility_summary(
            family=family, allowed_scene_count=9,
            requested_strata=("completion_enriched",),
            scene_evidence=[_scene_row(family, index) for index in range(5)],
            rejection_counts={"completion_snapshot_goal_claimed": 2})
        completion = summary["per_stratum"]["completion_enriched"]
        assert completion["required_distinct_scenes"] == 5
        assert completion["eligible_distinct_scenes"] == 5
        assert completion["quota_pass"] is True
        assert summary["all_requested_quotas_pass"] is True


def test_general_and_safety_hop_requirements_remain_frozen():
    selection = B.CORPUS_SELECTION_CONTRACT
    assert selection["strata"]["general"].endswith("graph_edges >= 2")
    assert selection["strata"]["safety_enriched"].endswith(
        "body-probe clearance <= 0.10m"
    )
    assert tuple(selection["state_selection_priority"]) == B.STRATA


def test_completed_failed_feasibility_gate_is_retained_without_rerun(tmp_path):
    census = _task_census()
    reductions = [
        _family_reduction(
            family, passed=index != 0, runtime_s=index + 0.5,
            census=census, out=tmp_path)
        for index, family in enumerate(B.STATE_SELECTOR.REQUIRED_FAMILIES)
    ]
    receipt = B.build_selector_feasibility_receipt_from_family_reductions(
        reductions=reductions, source=_source(),
        successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64, task_census=census)
    assert receipt["status"] == "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY"
    path = tmp_path / B.SELECTOR_FEASIBILITY_RECEIPT_NAME
    B.atomic_json(path, receipt)
    raw = path.read_bytes()
    loaded = B._load_completed_selector_feasibility(
        path, source=_source(),
        successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64, task_census=census)
    assert loaded == receipt
    assert path.read_bytes() == raw


def test_family_reduction_requires_exact_exhaustive_scene_coverage():
    family = B.STATE_SELECTOR.REQUIRED_FAMILIES[0]
    census = _task_census()
    reductions = [
        _family_reduction(name, census=census)
        for name in B.STATE_SELECTOR.REQUIRED_FAMILIES
    ]
    B.build_selector_feasibility_receipt_from_family_reductions(
        reductions=reductions, source=_source(),
        successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64, task_census=census)

    incomplete = {
        **reductions[0], "family_result": dict(reductions[0]["family_result"])}
    incomplete["family_result"]["scanned_scene_count"] = 6
    incomplete["family_reduction_digest"] = B.canonical_digest({
        key: value for key, value in incomplete.items()
        if key != "family_reduction_digest"
    })
    with pytest.raises(RuntimeError, match="is malformed"):
        B.build_selector_feasibility_receipt_from_family_reductions(
            reductions=[incomplete, *reductions[1:]], source=_source(),
            successor_selection_digest="b" * 64,
            exclusion_binding_digest="d" * 64, task_census=census)

    wrong_scene = {**reductions[0], "scene_shards": [
        dict(row) for row in reductions[0]["scene_shards"]]}
    wrong_scene["scene_shards"][0]["scene_task_digest"] = "e" * 64
    wrong_scene["family_reduction_digest"] = B.canonical_digest({
        key: value for key, value in wrong_scene.items()
        if key != "family_reduction_digest"
    })
    with pytest.raises(RuntimeError, match="is malformed"):
        B.build_selector_feasibility_receipt_from_family_reductions(
            reductions=[wrong_scene, *reductions[1:]], source=_source(),
            successor_selection_digest="b" * 64,
            exclusion_binding_digest="d" * 64, task_census=census)


def test_family_reduction_is_order_invariant_and_passes_frozen_gate():
    census = _task_census()
    reductions = [
        _family_reduction(family, runtime_s=index + 0.125, census=census)
        for index, family in enumerate(B.STATE_SELECTOR.REQUIRED_FAMILIES)
    ]
    forward = B.build_selector_feasibility_receipt_from_family_reductions(
        reductions=reductions, source=_source(),
        successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64, task_census=census)
    reverse = B.build_selector_feasibility_receipt_from_family_reductions(
        reductions=list(reversed(reductions)), source=_source(),
        successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64, task_census=census)
    assert reverse == forward
    assert forward["status"] == B.SELECTOR_FEASIBILITY_PASS_STATUS
    assert [row["family"] for row in forward["families"]] == \
        list(B.STATE_SELECTOR.REQUIRED_FAMILIES)
    assert set(forward["family_reduction_digests"]) == \
        set(B.STATE_SELECTOR.REQUIRED_FAMILIES)
    expected_lineage = [
        row for reduction in reductions for row in reduction["scene_shards"]
    ]
    assert forward["scene_shard_count"] == census["scene_task_count"]
    assert forward["scene_shard_lineage"] == expected_lineage
    assert forward["scene_shard_lineage_digest"] == \
        B.canonical_digest(expected_lineage)
    # This exercises the frozen V1 exhaustive reducer only.  Its accepted
    # failure receipt is predecessor evidence; the active V2 validator rightly
    # accepts only the new non-overwriting reachability receipt.
    B._verify_self_digest(
        forward, "state_selector_feasibility_receipt_digest",
        "legacy exhaustive selector receipt",
    )


def test_task_census_binds_manifest_and_genesis_scene_bytes(
        tmp_path):
    pool = {}
    for family in B.STATE_SELECTOR.REQUIRED_FAMILIES:
        scene = tmp_path / "train" / family / f"{family}-scene"
        scene.mkdir(parents=True)
        (scene / "manifest.json").write_text('{"frozen": true}\n')
        (scene / "genesis_scene.json").write_text('{"frozen": true}\n')
        pool[family] = [scene]
    first = B.build_selector_feasibility_task_census(
        pool=pool, source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64)
    second = B.build_selector_feasibility_task_census(
        pool={family: list(reversed(scenes)) for family, scenes in pool.items()},
        source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64)
    assert second == first
    first_task = first["families"][0]["tasks"][0]
    first_scene = pool[B.STATE_SELECTOR.REQUIRED_FAMILIES[0]][0]
    assert first_task["scene_manifest_sha256"] == \
        B.file_sha256(first_scene / "manifest.json")
    assert first_task["genesis_scene_sha256"] == \
        B.file_sha256(first_scene / "genesis_scene.json")
    B._validate_selector_feasibility_task_census(
        first, pool=pool, source=_source(),
        successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64)
    changed_scene = pool[B.STATE_SELECTOR.REQUIRED_FAMILIES[0]][0]
    (changed_scene / "genesis_scene.json").write_text('{"frozen": false}\n')
    with pytest.raises(RuntimeError, match="differs from the exact allow-list"):
        B._validate_selector_feasibility_task_census(
            first, pool=pool, source=_source(),
            successor_selection_digest="b" * 64,
            exclusion_binding_digest="d" * 64)


@pytest.mark.parametrize("name", ("sealed", "sealed_dev", "sealed_test.json"))
def test_scene_task_rejects_sealed_custody_path_before_file_read(
        tmp_path, monkeypatch, name):
    scene = tmp_path / name / "family" / "scene"
    monkeypatch.setattr(
        B, "file_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("sealed scene file was read")))
    with pytest.raises(RuntimeError, match="sealed benchmark paths"):
        B._selector_feasibility_scene_task("family", scene, 0)


def test_scene_pool_rejects_sealed_directory_before_traversal(
        tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "train" / "synthetic_blocked_family").mkdir(parents=True)
    invalid_index = SimpleNamespace(scene_ids=set(), binding=lambda: {})
    real_guard = B._assert_unsealed_path

    def synthetic_guard(path):
        if path.name == "synthetic_blocked_family":
            raise RuntimeError("synthetic sealed-path custody rejection")
        return real_guard(path)

    monkeypatch.setattr(B, "CORPUS", corpus)
    monkeypatch.setattr(B, "SPLITS", ("train",))
    monkeypatch.setattr(B, "_assert_unsealed_path", synthetic_guard)
    monkeypatch.setattr(B, "_factorial_scene_exclusions", lambda: (set(), {}))
    monkeypatch.setattr(B, "_pilot_scene_exclusions", lambda: (set(), {}))
    monkeypatch.setattr(
        B.INVALID_IDS, "load_invalid_identity_index", lambda: invalid_index)
    with pytest.raises(RuntimeError, match="sealed-path custody rejection"):
        B.scene_pool("scorer_fit")


def test_scene_task_rejects_symlink_to_unopened_sealed_target(
        tmp_path, monkeypatch):
    target = tmp_path / "sealed_synthetic_target"
    alias = tmp_path / "scene_alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        B, "file_sha256",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("symlink target was read")))
    with pytest.raises(RuntimeError, match="symlinked corpus paths"):
        B._selector_feasibility_scene_task(
            "family", alias / "family" / "scene", 0)


def test_scene_shard_is_nonbinding_and_process_exit_cannot_be_a_verdict():
    census = _task_census()
    task = B._selector_feasibility_family_tasks(
        census, B.STATE_SELECTOR.REQUIRED_FAMILIES[0])[0]
    shard = _scene_shard(task, census)
    assert shard["binding_receipt"] is False
    assert shard["eligibility_verdict_inferred_from_process_exit"] is False
    assert "verdict" not in shard["scene_result"]
    B._validate_selector_feasibility_scene_shard(
        shard, expected_task=task,
        expected_task_census_digest=census[
            "state_selector_feasibility_task_census_digest"],
        source=_source(), expected_successor_selection_digest="b" * 64,
        expected_exclusion_binding_digest="d" * 64)


@pytest.mark.parametrize("mode", ("missing", "invalid"))
def test_resume_reuses_exact_scene_shards_and_launches_only_missing_or_invalid(
        tmp_path, monkeypatch, mode):
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    out = tmp_path / "scorer_fit"
    out.mkdir()
    census = _task_census()
    family = B.STATE_SELECTOR.REQUIRED_FAMILIES[0]
    tasks = B._selector_feasibility_family_tasks(census, family)
    missing = tasks[2]
    original_bytes = {}
    for task in tasks:
        if task == missing:
            if mode == "invalid":
                B.atomic_json(
                    B._selector_feasibility_scene_shard_path(out, task),
                    {"invalid": True})
            continue
        path = B._selector_feasibility_scene_shard_path(out, task)
        B.atomic_json(path, _scene_shard(task, census))
        original_bytes[task["scene_id"]] = path.read_bytes()
    launched = []

    def launch(_args, task):
        launched.append(task["scene_task_digest"])
        B.atomic_json(
            B._selector_feasibility_scene_shard_path(out, task),
            _scene_shard(task, census))
        return 0

    monkeypatch.setattr(
        B, "_run_selector_feasibility_scene_subprocess", launch)
    shards = B._collect_selector_feasibility_scene_shards(
        args=SimpleNamespace(backend="cpu"), out=out, tasks=tasks,
        task_census_digest=census[
            "state_selector_feasibility_task_census_digest"],
        source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64)
    assert launched == [missing["scene_task_digest"]]
    assert len(shards) == len(tasks)
    for task in tasks:
        if task == missing:
            continue
        assert B._selector_feasibility_scene_shard_path(
            out, task).read_bytes() == original_bytes[task["scene_id"]]
    if mode == "invalid":
        assert len(list((out / "invalid_attempts").iterdir())) == 1


@pytest.mark.parametrize("mode", ("missing", "invalid"))
def test_post_outcome_missing_or_invalid_scene_shard_is_never_regenerated(
        tmp_path, monkeypatch, mode):
    out = tmp_path / "scorer_fit"
    out.mkdir()
    census = _task_census()
    task = B._selector_feasibility_family_tasks(
        census, B.STATE_SELECTOR.REQUIRED_FAMILIES[0])[0]
    path = B._selector_feasibility_scene_shard_path(out, task)
    original = None
    if mode == "invalid":
        B.atomic_json(path, {"invalid": True})
        original = path.read_bytes()
    monkeypatch.setattr(B, "_outcome_generation_started", lambda _out: True)
    monkeypatch.setattr(
        B, "_run_selector_feasibility_scene_subprocess",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("post-outcome scene worker was launched")))
    with pytest.raises(RuntimeError, match=f"is {mode} after outcomes"):
        B._collect_selector_feasibility_scene_shards(
            args=SimpleNamespace(backend="cpu"), out=out, tasks=[task],
            task_census_digest=census[
                "state_selector_feasibility_task_census_digest"],
            source=_source(), successor_selection_digest="b" * 64,
            exclusion_binding_digest="d" * 64)
    if mode == "missing":
        assert not path.exists()
    else:
        assert path.read_bytes() == original


def test_scene_sigsegv_preserves_prior_exact_scenes_and_records_no_ineligibility(
        tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    out = tmp_path / "scorer_fit"
    out.mkdir()
    census = _task_census()
    tasks = B._selector_feasibility_family_tasks(
        census, B.STATE_SELECTOR.REQUIRED_FAMILIES[0])
    first_path = B._selector_feasibility_scene_shard_path(out, tasks[0])
    B.atomic_json(first_path, _scene_shard(tasks[0], census))
    first_raw = first_path.read_bytes()
    monkeypatch.setattr(
        B, "_run_selector_feasibility_scene_subprocess",
        lambda _args, _task: -11)
    with pytest.raises(
            RuntimeError, match=r"exited -11.*no eligibility conclusion"):
        B._collect_selector_feasibility_scene_shards(
            args=SimpleNamespace(backend="cpu"), out=out, tasks=tasks,
            task_census_digest=census[
                "state_selector_feasibility_task_census_digest"],
            source=_source(), successor_selection_digest="b" * 64,
            exclusion_binding_digest="d" * 64)
    assert first_path.read_bytes() == first_raw
    assert not B._selector_feasibility_scene_shard_path(out, tasks[1]).exists()


def test_valid_atomic_scene_census_survives_worker_teardown_sigsegv(
        tmp_path, monkeypatch):
    monkeypatch.setattr(B, "OUT_ROOT", tmp_path)
    out = tmp_path / "scorer_fit"
    out.mkdir()
    census = _task_census()
    task = B._selector_feasibility_family_tasks(
        census, B.STATE_SELECTOR.REQUIRED_FAMILIES[0])[0]

    def write_then_crash(_args, exact_task):
        B.atomic_json(
            B._selector_feasibility_scene_shard_path(out, exact_task),
            _scene_shard(exact_task, census))
        return -11

    monkeypatch.setattr(
        B, "_run_selector_feasibility_scene_subprocess", write_then_crash)
    shards = B._collect_selector_feasibility_scene_shards(
        args=SimpleNamespace(backend="cpu"), out=out, tasks=[task],
        task_census_digest=census[
            "state_selector_feasibility_task_census_digest"],
        source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64)
    assert len(shards) == 1
    assert shards[0]["task"] == task
    assert shards[0]["eligibility_verdict_inferred_from_process_exit"] is False


def test_scene_reducer_is_order_invariant_and_preserves_family_failure():
    census = _task_census()
    family = B.STATE_SELECTOR.REQUIRED_FAMILIES[0]
    tasks = B._selector_feasibility_family_tasks(census, family)
    passing = [_scene_shard(task, census) for task in tasks]
    forward = B._reduce_selector_feasibility_family_scene_shards(
        family=family, tasks=tasks, shards=passing,
        task_census_digest=census[
            "state_selector_feasibility_task_census_digest"],
        source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64)
    reverse = B._reduce_selector_feasibility_family_scene_shards(
        family=family, tasks=tasks, shards=list(reversed(passing)),
        task_census_digest=census[
            "state_selector_feasibility_task_census_digest"],
        source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64)
    assert reverse == forward
    failing = [
        _scene_shard(task, census, include_completion=index < 4)
        for index, task in enumerate(tasks)
    ]
    failed = B._reduce_selector_feasibility_family_scene_shards(
        family=family, tasks=tasks, shards=failing,
        task_census_digest=census[
            "state_selector_feasibility_task_census_digest"],
        source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64)
    assert failed["family_result"]["verdict"] == "FAIL"


def test_production_scene_worker_writes_before_releasing_native_state(
        tmp_path, monkeypatch):
    census = _task_census()
    task = B._selector_feasibility_family_tasks(
        census, B.STATE_SELECTOR.REQUIRED_FAMILIES[0])[0]
    path = tmp_path / "scene.json"
    events = []
    references = {}

    class NativeObject:
        pass

    def load_shared(_backend):
        shared = NativeObject()
        references["shared"] = weakref.ref(shared)
        return shared

    def build_context(_scene_dir, *, seed, backend, shared):
        assert seed == task["drive_seed"] and backend == "cpu"
        assert shared is references["shared"]()
        ctx = NativeObject()
        references["ctx"] = weakref.ref(ctx)
        return ctx

    def scan_scene(*, ctx, **_kwargs):
        assert ctx is references["ctx"]()
        events.append("scan")
        return _scene_shard(task, census)["scene_result"]

    real_atomic_json = B.atomic_json

    def write_scene(write_path, payload):
        assert references["ctx"]() is not None
        assert references["shared"]() is not None
        events.append("write")
        real_atomic_json(write_path, payload)

    monkeypatch.setattr(B.V1, "_load_shared", load_shared)
    monkeypatch.setattr(B.V1, "build_context", build_context)
    monkeypatch.setattr(B, "_scan_selector_scene", scan_scene)
    monkeypatch.setattr(B, "atomic_json", write_scene)
    monkeypatch.setattr(B.gc, "collect", lambda: events.append("gc") or 0)
    payload = B._execute_selector_feasibility_scene_worker(
        args=SimpleNamespace(backend="cpu"), task=task, path=path,
        task_census_digest=census[
            "state_selector_feasibility_task_census_digest"],
        source=_source(), successor_selection_digest="b" * 64,
        exclusion_binding_digest="d" * 64)
    assert path.is_file()
    assert events == ["scan", "write", "gc"]
    assert payload["task"] == task


def test_scene_subprocess_command_is_exact_and_uses_no_shell(monkeypatch):
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=-11)

    monkeypatch.setattr(B.subprocess, "run", run)
    task = _task(B.STATE_SELECTOR.REQUIRED_FAMILIES[0], 2)
    assert B._run_selector_feasibility_scene_subprocess(
        SimpleNamespace(backend="cpu"), task) == -11
    assert observed["command"] == [
        B.sys.executable, str(B.Path(B.__file__).resolve()),
        "--pool", "scorer_fit", "--stage", "selector-feasibility",
        "--family", task["family"],
        "--selector-scene-id", task["scene_id"], "--backend", "cpu",
    ]
    assert observed["kwargs"]["check"] is False
    assert "shell" not in observed["kwargs"]


def test_binding_parent_reduces_shards_without_loading_genesis_and_is_idempotent(
        tmp_path, monkeypatch):
    out_root = tmp_path / "corpus"
    exclusion = {"strict_allow_list": True}
    exclusion_digest = B.canonical_digest(exclusion)
    census = _task_census(exclusion_digest=exclusion_digest)
    scorer_out = out_root / "scorer_fit"
    reductions = [
        _family_reduction(
            family, exclusion_digest=exclusion_digest, census=census,
            out=scorer_out)
        for family in B.STATE_SELECTOR.REQUIRED_FAMILIES
    ]
    monkeypatch.setattr(B, "OUT_ROOT", out_root)
    monkeypatch.setattr(B.STATE_SELECTOR, "validate_authority_artifacts", lambda: None)
    monkeypatch.setattr(
        B.STATE_SELECTOR, "validate_state_selector_feasibility_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        B.STATE_SELECTOR.PREDECESSOR,
        "validate_state_selector_feasibility_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(B, "clean_source_binding", _source)
    monkeypatch.setattr(B, "selection_digest", lambda: "b" * 64)
    monkeypatch.setattr(B, "scene_pool", lambda _pool: ({
        family: [] for family in B.STATE_SELECTOR.REQUIRED_FAMILIES
    }, exclusion))
    monkeypatch.setattr(
        B, "_issue_selector_feasibility_task_census",
        lambda **_kwargs: census)
    monkeypatch.setattr(
        B.V1, "_load_shared",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("binding reducer loaded Genesis in parent")))
    calls = []
    monkeypatch.setattr(
        B, "_reduce_selector_feasibility_families",
        lambda **_kwargs: calls.append("reduce") or reductions)
    args = SimpleNamespace(
        pool="scorer_fit", backend="cpu", family=None, stratum=None,
        selector_scene_id=None)
    assert B.stage_selector_feasibility(args) == 0
    assert calls == ["reduce"]
    receipt_path = out_root / "scorer_fit" / B.SELECTOR_FEASIBILITY_RECEIPT_NAME
    first = receipt_path.read_bytes()

    monkeypatch.setattr(
        B, "_reduce_selector_feasibility_families",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("valid complete receipt was reduced again")))
    assert B.stage_selector_feasibility(args) == 0
    assert receipt_path.read_bytes() == first
    assert B._load_completed_selector_feasibility(
        receipt_path, source=_source(),
        successor_selection_digest="b" * 64,
        exclusion_binding_digest=exclusion_digest,
        task_census=census) is not None
    changed_source = {**_source(), "bound_implementations_digest": "e" * 64}
    assert B._load_completed_selector_feasibility(
        receipt_path, source=changed_source,
        successor_selection_digest="b" * 64,
        exclusion_binding_digest=exclusion_digest,
        task_census=census) is None
    assert B._load_completed_selector_feasibility(
        receipt_path, source=_source(),
        successor_selection_digest="b" * 64,
        exclusion_binding_digest="f" * 64,
        task_census=census) is None
    first_task = B._selector_feasibility_family_tasks(
        census, B.STATE_SELECTOR.REQUIRED_FAMILIES[0])[0]
    B.atomic_json(
        B._selector_feasibility_scene_shard_path(scorer_out, first_task),
        {"invalid": True})
    assert B._load_completed_selector_feasibility(
        receipt_path, source=_source(),
        successor_selection_digest="b" * 64,
        exclusion_binding_digest=exclusion_digest,
        task_census=census) is None


def test_preserved_state_identity_is_exact_and_cannot_be_rewrapped():
    state = next(iter(B._preserved_states_by_digest().values()))
    assert B._state_identity_matches_active_or_preserved(dict(state)) is True
    changed = dict(state)
    changed["warmup_blocks"] = int(changed["warmup_blocks"]) + 1
    assert B._state_identity_matches_active_or_preserved(changed) is False
