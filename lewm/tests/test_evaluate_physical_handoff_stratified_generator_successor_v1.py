from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from lewm.safety import physical_handoff_stratified_generator_successor_v1_contract as C
from lewm.safety import physical_handoff_stratified_generator_successor_v1_metrics as M
from scripts import evaluate_physical_handoff_stratified_generator_successor_v1 as E


def _binding(path: str, marker: str) -> dict:
    return {
        "path": path,
        "bytes": 1,
        "sha256": marker * 64,
        "nlink": 1,
        "ordinary_regular_file": True,
        "resolved_path_ancestor_symlink_count": 0,
    }


def _available_records() -> list[dict]:
    flags = {name: False for name in C.TERMINATION_FLAGS_FIELDS}
    criteria = {
        name: True for name in C.TEACHER_QUALIFICATION_COMPONENT_IDS
    }
    records: list[dict] = []
    # Candidate-index order is the frozen round-major registry order.  Every
    # stream reaches its fourth qualified state at attempt three.
    for attempt in range(C.TARGET_QUALIFIED_PER_STREAM):
        for family in C.FAMILY_IDS:
            for stratum in range(C.STRATA_PER_FAMILY):
                spec = C.build_candidate_spec(family, stratum, attempt)
                state = M.build_state_disposition_record(
                    spec,
                    stage_reached="COMPLETE",
                    initial_termination_flags=flags,
                    probe_trial_termination_flags=[flags, flags],
                    teacher_termination_flags=flags,
                    probe_tip_sample_indices=[None, None],
                    teacher_criteria=criteria,
                    executable_snapshot_exists=True,
                    teacher_executed=True,
                    snapshot_identity={
                        "artifact_file_sha256": "a" * 64,
                        "snapshot_semantic_digest_v1": "b" * 64,
                        "snapshot_behavioural_digest_v1": "c" * 64,
                    },
                )
                prefix = (
                    f"qualification/{spec['stream_id']}/attempt-{attempt:02d}"
                )
                records.append(
                    M.build_generator_terminal_record(
                        state,
                        source_freeze_commit="d" * 40,
                        runtime_contract_content_digest="e" * 64,
                        material_metadata_binding=_binding(
                            f"{prefix}/metadata.json", "f"
                        ),
                        material_payload_binding=_binding(
                            f"{prefix}/payload.npz", "0"
                        ),
                        persisted_array_evidence_sha256="1" * 64,
                    )
                )
    return records


def _initial_tipped_record(spec: dict) -> dict:
    flags = {name: name == "tipped" for name in C.TERMINATION_FLAGS_FIELDS}
    state = M.build_state_disposition_record(
        spec,
        stage_reached="INITIAL_BOUNDARY",
        initial_termination_flags=flags,
        executable_snapshot_exists=False,
        teacher_executed=False,
        diagnostics_inventory=tuple(sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY)),
        payload_member_inventory=tuple(sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY)),
    )
    prefix = (
        f"qualification/{spec['stream_id']}/"
        f"attempt-{spec['attempt_index']:02d}"
    )
    return M.build_generator_terminal_record(
        state,
        source_freeze_commit="d" * 40,
        runtime_contract_content_digest="e" * 64,
        material_metadata_binding=_binding(f"{prefix}/metadata.json", "2"),
        material_payload_binding=_binding(f"{prefix}/payload.npz", "3"),
        persisted_array_evidence_sha256="4" * 64,
    )


def test_independent_generator_reducer_matches_pure_authority_exactly() -> None:
    records = _available_records()
    expected = M.build_generator_metrics(records)
    observed = E._independent_generator_metrics(records)
    assert E.canonical_bytes(observed) == E.canonical_bytes(expected)
    assert observed["status"] == C.GENERATOR_PANEL_AVAILABLE
    assert observed["terminal_record_count"] == 256
    assert len(observed["selected_candidate_indices"]) == 64
    assert observed["v4_shortfall_resolution"]["conclusion"] == (
        C.V4_SHORTFALL_SUPPORTS_SHALLOW_SAMPLING
    )
    assert observed["v4_shortfall_resolution"][
        "all_canonical_v4_shortfalls_resolved"
    ] is True


def test_independent_generator_reducer_rejects_stream_after_fourth_qualified() -> None:
    records = _available_records()
    duplicate = copy.deepcopy(records[-1])
    duplicate["candidate_index"] = C.candidate_index(
        C.FAMILY_IDS[-1], C.STRATA_PER_FAMILY - 1, 4
    )
    duplicate["attempt_index"] = 4
    duplicate["qualified"] = False
    records.append(duplicate)
    with pytest.raises(E.RegenerationError, match="continued beyond fourth"):
        E._independent_generator_metrics(records)


@pytest.mark.parametrize(
    ("qualified_prefix", "expected_status", "expected_shortfall_conclusion"),
    [
        (
            0,
            C.GENERATOR_FEASIBILITY_NO_GO,
            C.GENERATOR_FEASIBILITY_NO_GO,
        ),
        (1, C.GENERATOR_LOW_YIELD, C.V4_SHORTFALL_INSUFFICIENT),
    ],
)
def test_independent_generator_reducer_exact_failure_classification(
    qualified_prefix: int,
    expected_status: str,
    expected_shortfall_conclusion: str,
) -> None:
    family = "TURNING_JUNCTION"
    stratum = 0
    records = [
        row
        for row in _available_records()
        if not (row["family"] == family and row["stratum_index"] == stratum)
    ]
    if qualified_prefix:
        records.extend(
            row
            for row in _available_records()
            if row["family"] == family
            and row["stratum_index"] == stratum
            and row["attempt_index"] < qualified_prefix
        )
    records.extend(
        _initial_tipped_record(C.build_candidate_spec(family, stratum, attempt))
        for attempt in range(qualified_prefix, C.MAX_ATTEMPTS_PER_STREAM)
    )
    records.sort(key=lambda row: row["candidate_index"])
    observed = E._independent_generator_metrics(records)
    expected = M.build_generator_metrics(records)
    assert E.canonical_bytes(observed) == E.canonical_bytes(expected)
    assert observed["status"] == expected_status
    assert observed["next_decision"] == C.TURNING_JUNCTION_GENERATOR_NEXT_DECISION
    assert observed["v4_shortfall_resolution"]["conclusion"] == (
        expected_shortfall_conclusion
    )


def _independent_success_decision_fixture() -> tuple[dict, dict]:
    states = []
    fanout = []
    heldout = []
    for panel_index in range(C.PANEL_STATE_COUNT):
        heldout_role = panel_index % 4 == 0
        family = C.FAMILY_IDS[panel_index // C.STRATA_PER_FAMILY]
        state_id = f"state-{panel_index:02d}"
        states.append(
            {
                "panel_index": panel_index,
                "state_id": state_id,
                "family": family,
                "role": "DEVELOPMENT_HELDOUT" if heldout_role else "DEVELOPMENT",
            }
        )
        for branch_index in range(len(C.CANDIDATE_IDS)):
            fanout.append(
                {
                    "state_id": state_id,
                    "branch_candidate_index": branch_index,
                    "oracle_admissible": True,
                    "entered_correct_edge": True,
                    "successor_viable": True,
                    "positive_port_progress": True,
                    "physics_contact": False,
                    "stuck": False,
                    "command_tracking_rows": [
                        {
                            "post_slew_command": [0.0, 0.0, 0.0],
                            "mean_achieved_body_velocity": [0.0, 0.0, 0.0],
                            "active_vx": False,
                            "active_yaw": False,
                        }
                        for _ in range(C.HORIZON_TICKS[C.PRIMARY_HORIZON])
                    ],
                }
            )
        if heldout_role:
            for condition in C.HELDOUT_CONDITION_IDS:
                heldout.append(
                    {
                        "state_id": state_id,
                        "family": family,
                        "condition_id": condition,
                        "selected_candidate_index": (
                            None if condition == "TEACHER_TRACE" else 0
                        ),
                        "teacher_correct_execution": (
                            True if condition == "TEACHER_TRACE" else None
                        ),
                        "correct_edge_top1": (
                            None if condition == "TEACHER_TRACE" else True
                        ),
                        "correct_edge_top3": (
                            None if condition == "TEACHER_TRACE" else True
                        ),
                        "selected_correct_edge_execution": (
                            None if condition == "TEACHER_TRACE" else True
                        ),
                        "normalized_port_regret": (
                            None if condition == "TEACHER_TRACE" else 0.0
                        ),
                    }
                )
    summaries = [
        {
            "target_id": target,
            "selected_correct_edge_execution_rate": 1.0,
            "correct_edge_top3_rate": 1.0,
            "normalized_port_regret": 0.0,
            "mean_selected_port_progress_m": 1.0,
        }
        for target in C.TARGET_IDS
    ]
    tracking = {
        "selected_branch_count": 32,
        "command_tick_count": 480,
        "active_vx_component_count": 0,
        "active_yaw_component_count": 0,
        "vx_mae_mps": 0.0,
        "vy_absolute_mean_mps": 0.0,
        "yaw_rate_mae_rad_s": 0.0,
        "commanded_sign_agreement_rate": 1.0,
        "passed": True,
    }
    classification_input = {
        "teacher_correct_execution_count": 16,
        "coverage_rate": 1.0,
        "ranker_correct_edge_top1_rate": 1.0,
        "ranker_correct_edge_top3_rate": 1.0,
        "ranker_selected_correct_edge_execution_rate": 1.0,
        "ranker_normalized_port_regret": 0.0,
        "oracle_selected_correct_edge_execution_rate": 1.0,
        "oracle_covered_state_correct_execution_rate": 1.0,
        "repeatability_rate": 1.0,
        "command_tracking_pass": True,
        "minimum_family_correct_execution_count": 4,
        "selected_target_id": "TARGET_NODE_CENTRE",
        "selected_target_passes_handoff_gate": True,
        "selected_target_materially_outperforms_node_centre": False,
    }
    decision = E._independent_classification(classification_input)
    metrics = {
        "generator_status": C.GENERATOR_PANEL_AVAILABLE,
        "primary_classification": decision["primary_classification"],
        "next_decision": decision["next_decision"],
        "runtime_environments": {
            "encoder": {
                "checkpoint_sha256": C.FROZEN_ENCODER_CHECKPOINT_SHA256
            },
            "ranker": {
                "checkpoint_sha256": C.FROZEN_RANKER_CHECKPOINT_SHA256
            },
        },
        "downstream": {
            "heldout": {
                "comparator_alias_authority": copy.deepcopy(
                    C.HELDOUT_COMPARATOR_ALIAS_AUTHORITY
                )
            },
            "classification_input": classification_input,
            "command_tracking": tracking,
            "gate": {"authority": copy.deepcopy(C.HANDOFF_GATE), "passed": True},
            "component_failures": decision["component_failures"],
            "active_components_in_precedence_order": decision[
                "active_components_in_precedence_order"
            ],
            "earliest_failing_component": decision["earliest_failing_component"],
        },
    }
    evidence = {
        "panel_manifest.json": {"states": states},
        "development_target_selection.json": {
            "selected_target_id": "TARGET_NODE_CENTRE",
            "target_summaries": summaries,
        },
        "candidate_fanout.jsonl": fanout,
        "heldout_scores.jsonl": heldout,
        "repeatability.jsonl": [{"repeat_success": True} for _ in range(64)],
    }
    return metrics, evidence


def test_evaluator_independently_reduces_decisive_downstream_gate() -> None:
    metrics, evidence = _independent_success_decision_fixture()
    E._independent_validate_downstream_decision(metrics, evidence)
    tampered = copy.deepcopy(metrics)
    tampered["downstream"]["gate"]["passed"] = False
    with pytest.raises(E.RegenerationError, match="independent evaluator reduction"):
        E._independent_validate_downstream_decision(tampered, evidence)

    alias_tampered = copy.deepcopy(metrics)
    alias_tampered["downstream"]["heldout"]["comparator_alias_authority"][
        "user_facing_to_internal_condition_id"
    ]["PHYSICAL_TEACHER"] = "PHYSICAL_TEACHER"
    alias_tampered["downstream"]["heldout"]["comparator_alias_authority"] = (
        C.attach_content_digest(
            {
                key: value
                for key, value in alias_tampered["downstream"]["heldout"][
                    "comparator_alias_authority"
                ].items()
                if key != "content_digest"
            }
        )
    )
    with pytest.raises(E.RegenerationError, match="comparator alias authority drift"):
        E._independent_validate_downstream_decision(alias_tampered, evidence)


def test_evaluator_independent_classification_rejects_false_gate_projection() -> None:
    metrics, _evidence = _independent_success_decision_fixture()
    classification = copy.deepcopy(metrics["downstream"]["classification_input"])
    classification["selected_target_passes_handoff_gate"] = False
    with pytest.raises(E.RegenerationError, match="full-handoff gate"):
        E._independent_classification(classification)


def test_source_observation_is_freeze_only_before_and_after_empty_result_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = "a" * 40
    result = "b" * 40
    tree = "c" * 40
    relative = "tracked.py"
    source = b"value = 1\n"
    (tmp_path / relative).write_bytes(source)
    closure_path = tmp_path / (
        f"{C.DOC_PREFIX}_source_closure_2026-09-04.json"
    )
    closure_path.parent.mkdir(parents=True)
    closure = C.attach_content_digest(
        {
            "schema": f"{C.OUTPUT_BASENAME}.source_closure.v1",
            "rows": [
                {
                    "path": relative,
                    "bytes": len(source),
                    "sha256": hashlib.sha256(source).hexdigest(),
                }
            ],
        }
    )
    closure_path.write_bytes(E.canonical_bytes(closure))
    monkeypatch.setattr(E, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(C, "SOURCE_CLOSURE_PATHS", (relative,))
    monkeypatch.setattr(
        E.subprocess,
        "check_output",
        lambda arguments, **_kwargs: source
        if arguments == ["git", "show", f"{freeze}:{relative}"]
        else (_ for _ in ()).throw(AssertionError(arguments)),
    )

    observed_head = freeze

    def fake_git(*arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return observed_head
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments == ("show", "-s", "--format=%s", freeze):
            return C.CONTRACT_FREEZE_COMMIT_SUBJECT
        if arguments == ("rev-list", "--parents", "-n", "1", freeze):
            return f"{freeze} {C.SOURCE_PARENT_COMMIT}"
        if arguments == ("rev-parse", f"{freeze}^{{tree}}"):
            return tree
        if arguments == ("show", "-s", "--format=%s", result):
            return C.RESULT_COMMIT_SUBJECT
        if arguments == ("rev-list", "--parents", "-n", "1", result):
            return f"{result} {freeze}"
        if arguments == ("rev-parse", f"{result}^{{tree}}"):
            return tree
        if arguments == ("diff", "--name-only", freeze, result):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(E, "_git", fake_git)
    runtime = {"source_freeze_commit": freeze}
    at_freeze = E._observe_source_freeze(runtime)
    observed_head = result
    after_result = E._observe_source_freeze(runtime)
    assert E.canonical_bytes(at_freeze) == E.canonical_bytes(after_result)
    assert at_freeze["observed_head_commit_at_scientific_reduction"] == freeze


def test_evaluator_reopens_full_fanout_wrapper_for_repeat_validation() -> None:
    source = Path(E.__file__).read_text()
    material = source[
        source.index("def _validate_persisted_material") : source.index(
            "def build_reduction"
        )
    ]
    assert "source_fanout_material=fanout_validation" in material
    assert "source_fanout_material=fanout_metadata" not in material


@pytest.mark.parametrize(
    "suffix",
    (
        "_regeneration_receipt.json",
        "_custody_receipt.json",
        "_terminal_custody_bundle.json",
    ),
)
def test_evaluator_rejects_every_conventional_external_receipt_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    root = tmp_path / C.OUTPUT_BASENAME
    root.mkdir()
    paths = tuple(
        str(root.parent / f"{root.name}{value}")
        for value in (
            "_regeneration_receipt.json",
            "_custody_receipt.json",
            "_terminal_custody_bundle.json",
        )
    )
    authority = copy.deepcopy(C.PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY)
    authority.pop("content_digest")
    authority["paths"] = list(paths)
    monkeypatch.setattr(C, "PROHIBITED_EXTERNAL_PUBLICATION_PATHS", paths)
    monkeypatch.setattr(
        C,
        "PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY",
        C.attach_content_digest(authority),
    )
    sibling = root.parent / f"{root.name}{suffix}"
    sibling.write_bytes(b"unauthorized\n")
    with pytest.raises(E.RegenerationError, match="external regeneration/custody"):
        E._require_no_successor_external_receipt(root)


def test_report_and_existing_validation_apply_receipt_guard_before_and_after() -> None:
    source = Path(E.__file__).read_text()
    for start, end in (
        ("def reduce_publish_and_validate", "def validate_existing_publication"),
        ("def validate_existing_publication", "def build_parser"),
    ):
        body = source[source.index(start) : source.index(end)]
        assert body.count("_require_no_successor_external_receipt(root)") >= 2
