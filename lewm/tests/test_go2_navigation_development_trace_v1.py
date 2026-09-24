from __future__ import annotations

import hashlib

import pytest

from lewm.benchmarks.go2_navigation_development_trace_v1 import (
    ACTION_SOURCES,
    ActualOpenLedgerV1,
    CANONICAL_COLORS,
    CallCounterPanelV1,
    ControllerEpisodeBindingV1,
    ControllerTraceV1,
    EMPTY_TICK_CHAIN_SHA256,
    NavigationTickRecordV1,
    NavigationTraceV1HashError,
    NavigationTraceV1SchemaError,
    ResetReceiptV1,
    canonical_binary64_hex,
    canonical_json_bytes,
    canonical_json_sha256,
    decode_canonical_binary64_hex,
    parse_canonical_json_bytes,
    parse_controller_trace_v1,
    zero_owner_revisions,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _binding() -> ControllerEpisodeBindingV1:
    names = (
        "shared_v5_checkpoint_file_sha256",
        "shared_v5_model_state_sha256",
        "g2_report_sha256",
        "g2_candidate_publication_sha256",
        "target_head_checkpoint_sha256",
        "target_head_config_sha256",
        "target_head_calibration_sha256",
        "g4_head_checkpoint_sha256",
        "g4_head_config_sha256",
        "physical_calibration_sha256",
        "physical_thresholds_sha256",
        "geometry_profile_sha256",
        "runner_config_sha256",
        "controller_config_sha256",
        "follower_config_sha256",
        "captured_source_graph_sha256",
    )
    return ControllerEpisodeBindingV1(
        **{name: _h(name) for name in names},
        semantic_colors=CANONICAL_COLORS,
        tick_budget=8,
        execution_seed=20260714,
        reset_id="reset-001",
        session_id="session-001",
        authority_mode="synthetic_mock",
    )


def _reset(binding: ControllerEpisodeBindingV1) -> ResetReceiptV1:
    return ResetReceiptV1(
        binding_sha256=binding.content_sha256,
        reset_id=binding.reset_id,
        session_id=binding.session_id,
        reset_capability_id="capability-001",
        physical_memory_owner_id="owner-physical",
        configuration_projection_owner_id="owner-configuration",
        planner_owner_id="owner-planner",
        view_owner_id="owner-view",
        target_memory_owner_ids=(
            "owner-target-red",
            "owner-target-yellow",
            "owner-target-blue",
            "owner-target-green",
        ),
        router_owner_id="owner-router",
        follower_owner_id="owner-follower",
        integration_owner_id="owner-integration",
        action_journal_owner_id="owner-action",
        claim_journal_owner_id="owner-claim",
        trace_owner_id="owner-trace",
        owner_revisions=zero_owner_revisions(),
        empty_action_journal_sha256=_h("empty-action"),
        empty_claim_journal_sha256=_h("empty-claim"),
        empty_tick_chain_sha256=EMPTY_TICK_CHAIN_SHA256,
        reset_clearance_certificate_sha256=None,
    )


def _counts(*, g4: int) -> CallCounterPanelV1:
    return CallCounterPanelV1(1, 1, 1, 1, 1, g4, 1, 1, 0)


def _tick(reset: ResetReceiptV1) -> NavigationTickRecordV1:
    hashes = {
        name: _h(name)
        for name in (
            "controller_input_sha256",
            "inference_receipt_sha256",
            "pre_physical_content_sha256",
            "post_physical_content_sha256",
            "physical_transaction_sha256",
            "physical_retraction_sha256",
            "pre_configuration_content_sha256",
            "post_configuration_content_sha256",
            "configuration_snapshot_sha256",
            "configuration_component_sha256",
            "frontier_sha256",
            "tick_admission_receipt_sha256",
            "view_admission_sha256",
            "scheduler_rows_sha256",
            "waypoint_receipt_sha256",
            "follower_receipt_sha256",
            "requested_command_block_sha256",
            "executed_command_block_sha256",
            "platform_envelope_clipping_sha256",
            "broker_execution_sha256",
            "broker_fall_sha256",
        )
    }
    return NavigationTickRecordV1(
        tick_index=0,
        timestamp_binary64_hex=canonical_binary64_hex(1.25),
        synchronization_id="sync-000",
        reset_id=reset.reset_id,
        session_id=reset.session_id,
        **hashes,
        per_tick_counts=_counts(g4=1),
        cumulative_counts=_counts(g4=1),
        pre_physical_revision=0,
        post_physical_revision=1,
        pre_configuration_revision=0,
        post_configuration_revision=1,
        pre_view_revision=0,
        post_view_revision=1,
        target_outcome_kinds=("positive", "qualified_negative", "abstain", "abstain"),
        target_outcome_receipt_sha256s=tuple(_h(f"outcome-{index}") for index in range(4)),
        pre_target_revisions=(0, 0, 0, 0),
        post_target_revisions=(1, 1, 1, 1),
        posterior_sha256s=tuple(_h(f"posterior-{index}") for index in range(4)),
        posterior_component_sha256s=tuple(_h(f"component-{index}") for index in range(4)),
        posterior_ages=(0, 1, 1, 1),
        locked_color=None,
        decision_kind="exploration",
        target_route_sha256=None,
        g4_candidate_set_sha256=_h("candidate-set"),
        baseline_scores_sha256=_h("baseline-scores"),
        learned_scores_sha256=_h("learned-scores"),
        selected_row=2,
        selected_path_sha256=_h("selected-path"),
        terminal_yaw_binary64_hex=canonical_binary64_hex(0.5),
        action_source="learned_g4",
        claim_intent_sha256=None,
        pre_claim_journal_revision=0,
        post_claim_journal_revision=0,
        controller_fault_code=None,
        stall_state="moving",
        previous_tick_chain_sha256=reset.empty_tick_chain_sha256,
    )


def _trace() -> ControllerTraceV1:
    binding = _binding()
    reset = _reset(binding)
    tick = _tick(reset)
    projection = ActualOpenLedgerV1().append(
        actor="controller",
        phase="controller",
        role="captured_source",
        no_follow_canonical_path="/synthetic/source.py",
        expected_sha256=_h("source"),
        actual_sha256=_h("source"),
        access_disposition="allowed",
    )
    revisions = dict(zero_owner_revisions())
    for name in (
        "physical",
        "configuration",
        "view",
        "target_red",
        "target_yellow",
        "target_blue",
        "target_green",
        "tick_chain",
    ):
        revisions[name] = 1
    return ControllerTraceV1(
        episode_binding=binding,
        reset_receipt=reset,
        ticks=(tick,),
        semantic_claim_intent_sha256s=(),
        action_source_counts=tuple(
            (name, 1 if name == "learned_g4" else 0) for name in ACTION_SOURCES
        ),
        final_owner_revisions=tuple(
            (name, revisions[name]) for name, _ in zero_owner_revisions()
        ),
        terminal_status="completed",
        inference_counts=_counts(g4=1),
        evaluator_access_count=0,
        evaluator_callback_count=0,
        actual_open_controller_projection=projection,
    )


def test_closed_trace_round_trip_and_hash_chain() -> None:
    trace = _trace()
    raw = trace.to_canonical_bytes()
    restored = parse_controller_trace_v1(raw)
    assert restored.to_dict() == trace.to_dict()
    assert restored.ticks[0].previous_tick_chain_sha256 == EMPTY_TICK_CHAIN_SHA256
    assert restored.final_tick_chain_sha256 == restored.ticks[0].chain_sha256
    assert restored.inference_counts.observation_tick_count == 1
    assert restored.evaluator_access_count == 0


def test_canonical_json_rejects_duplicate_float_noncanonical_and_mapping_subclass() -> None:
    class DictSubclass(dict):
        pass

    with pytest.raises(NavigationTraceV1SchemaError):
        canonical_json_bytes(DictSubclass({"a": 1}))
    with pytest.raises(NavigationTraceV1SchemaError):
        parse_canonical_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(NavigationTraceV1SchemaError):
        parse_canonical_json_bytes(b'{"a":1.5}')
    with pytest.raises(NavigationTraceV1SchemaError):
        parse_canonical_json_bytes(b'{ "a":1}')
    with pytest.raises(NavigationTraceV1SchemaError):
        canonical_json_bytes({1: "nonstring"})


def test_binary64_encoding_is_finite_and_canonical() -> None:
    assert decode_canonical_binary64_hex(canonical_binary64_hex(1.5)) == 1.5
    assert canonical_binary64_hex(-0.0) == "0000000000000000"
    with pytest.raises(NavigationTraceV1SchemaError):
        decode_canonical_binary64_hex("8000000000000000")
    with pytest.raises(NavigationTraceV1SchemaError):
        canonical_binary64_hex(float("nan"))


def test_nested_extra_key_and_commitment_mutation_reject() -> None:
    binding = _binding()
    extra = binding.to_dict()
    extra["scene_id"] = "forbidden"
    with pytest.raises(NavigationTraceV1SchemaError):
        ControllerEpisodeBindingV1.from_dict(extra)
    mutated = binding.to_dict()
    mutated["tick_budget"] = 9
    with pytest.raises(NavigationTraceV1HashError):
        ControllerEpisodeBindingV1.from_dict(mutated)


def test_ledger_rejects_path_escape_duplicate_and_wrong_allowed_hash() -> None:
    ledger = ActualOpenLedgerV1()
    with pytest.raises(NavigationTraceV1SchemaError):
        ledger.append(
            actor="controller",
            phase="controller",
            role="source",
            no_follow_canonical_path="/synthetic/../escape.py",
            expected_sha256=_h("a"),
            actual_sha256=_h("a"),
            access_disposition="allowed",
        )
    with pytest.raises(NavigationTraceV1SchemaError):
        ledger.append(
            actor="controller",
            phase="controller",
            role="source",
            no_follow_canonical_path="/synthetic/source.py",
            expected_sha256=_h("a"),
            actual_sha256=_h("b"),
            access_disposition="allowed",
        )
    row = ledger.append(
        actor="controller",
        phase="controller",
        role="source",
        no_follow_canonical_path="/synthetic/source.py",
        expected_sha256=_h("a"),
        actual_sha256=_h("a"),
        access_disposition="allowed",
    )
    with pytest.raises(NavigationTraceV1SchemaError):
        row.append(
            actor="controller",
            phase="controller",
            role="source",
            no_follow_canonical_path="/synthetic/source.py",
            expected_sha256=_h("a"),
            actual_sha256=_h("a"),
            access_disposition="allowed",
        )


def test_trace_rejects_noncontiguous_tick_and_evaluator_projection() -> None:
    trace = _trace()
    bad_tick = trace.ticks[0].to_dict()
    bad_tick["tick_index"] = 1
    core = dict(bad_tick)
    core.pop("content_sha256")
    core.pop("chain_sha256")
    bad_tick["content_sha256"] = canonical_json_sha256(core)
    bad_tick["chain_sha256"] = canonical_json_sha256(
        {
            "schema": "lewm_go2_navigation_tick_chain_link_v1",
            "version": 1,
            "previous_tick_chain_sha256": bad_tick["previous_tick_chain_sha256"],
            "tick_content_sha256": bad_tick["content_sha256"],
        }
    )
    tick = NavigationTickRecordV1.from_dict(bad_tick)
    with pytest.raises(NavigationTraceV1SchemaError):
        ControllerTraceV1(
            episode_binding=trace.episode_binding,
            reset_receipt=trace.reset_receipt,
            ticks=(tick,),
            semantic_claim_intent_sha256s=(),
            action_source_counts=trace.action_source_counts,
            final_owner_revisions=trace.final_owner_revisions,
            terminal_status="completed",
            inference_counts=trace.inference_counts,
            evaluator_access_count=0,
            evaluator_callback_count=0,
            actual_open_controller_projection=trace.actual_open_controller_projection,
        )

