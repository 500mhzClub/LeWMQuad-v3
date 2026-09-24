from types import SimpleNamespace
from copy import deepcopy
import pytest
from scripts import run_go2_all_phase_residual_maze02_matched_native_v1 as mod
from scripts import all_phase_residual_maze02_native_inputs_development as inputs


@pytest.mark.parametrize('fault',['none','different_state','different_condition','different_variant','different_seed'])
def test_exact_assigned_corrected_models(monkeypatch,fault):
    case=mod.CASES[0];model=SimpleNamespace(state_dict=lambda:{})
    monkeypatch.setattr(mod,'load_assigned',lambda *a:(model,
        'direct' if fault=='different_condition' else case[3],
        'no_rgb' if fault=='different_variant' else case[2]))
    monkeypatch.setattr(mod,'state_digest',lambda _: 'changed' if fault=='different_state' else 'assigned')
    launch=dict(input_admission={'correction_admission':{}},assigned_model_states={case[4]:'assigned'})
    if fault=='different_seed':case=(*case[:4],'seed_2026091401_full_jepa')
    if fault=='none':assert mod.assigned_model(launch,case) is model
    else:
        with pytest.raises(ValueError,match='assigned'):mod.assigned_model(launch,case)


@pytest.mark.parametrize('owner',[inputs.NATIVE_OWNER,inputs.CORRECTION_OWNER])
def test_live_original_owner_blocks_input_admission(monkeypatch,owner):
    monkeypatch.setattr(inputs.corrected_wait,'owner_live',lambda candidate:candidate==owner)
    monkeypatch.setattr(inputs,'completed',lambda *a:pytest.fail('read a pending result'))
    with pytest.raises(ValueError,match='owners must finish'):inputs.admit_inputs('wait','native',{})


def native_fixture():
    prefix=dict(physical_and_public_prefix_exact=True,all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,candidate_intervention_command_completed=True,
        common_prefix_frames=1207,first_intervention_frame=1206,physical_prefix_samples=61050)
    report=dict(raw_sensor_reconstruction_pass=True,raw_model_command_replay_pass=True,raw_command_audit_pass=True,
        model_state_unchanged=True,verified_round_trip=False,native_evaluation={},strict_physical_visibility_pass=False,
        hard_measurement_failed_frames=[8],renderer_capture_audit={})
    terminal=dict(status='RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_COLLECTED_AND_RAW_AUDITED',case=inputs.native.CASE[0],
        layout_index=3,prefix_comparison=deepcopy(prefix),model_state_unchanged=True,artifact_sha256={'leaf':'sha'},
        **{k:report[k] for k in mod.OUTCOME_KEYS})
    result=dict(status='RECENT_QUALIFIED_DIRECT_FLOW_MAZE03_PILOT_V1_COMPLETE',conditions=[deepcopy(terminal)],
        measured_round_trip_successes=0,artifact_sha256={'leaf':'sha'})
    return result,dict(planned_case=list(inputs.native.CASE)),report,prefix,terminal


def test_completed_native_failure_is_not_promoted_or_bypassed():
    result=inputs.admit_native(*native_fixture())
    assert result['measured_round_trip_successes']==0 and result['scientific_success_required'] is False


@pytest.mark.parametrize('fault',['raw','missing_artifact','terminal','prefix','count'])
def test_original_native_evidence_must_be_complete(fault):
    args=list(native_fixture())
    if fault=='raw':args[2]['raw_model_command_replay_pass']=False
    if fault=='missing_artifact':args[0]['artifact_sha256']={}
    if fault=='terminal':args[4]['failure']='failed'
    if fault=='prefix':args[3]['candidate_intervention_command_completed']=False
    if fault=='count':args[0]['measured_round_trip_successes']=1
    with pytest.raises(ValueError):inputs.admit_native(*args)
