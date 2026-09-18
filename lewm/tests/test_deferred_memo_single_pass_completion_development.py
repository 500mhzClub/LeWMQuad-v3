"""Complete timing populations, exact state witnesses and ended-owner checks."""
from copy import deepcopy
import pytest
from scripts import verify_go2_deferred_memo_single_pass_completion_v1 as check
from lewm.tests.test_deferred_memo_single_pass_runner_development import prior as prior_report


def payload():
    reference_rows=[];rows=[]
    for frame in range(1428):
        identity=f'{frame:064x}'
        reference_rows.append(dict(public_input_sha256=identity,original_decision_sha256=identity,candidate_decision_sha256=identity))
        rows.append(dict(frame=frame,public_input_sha256=identity,original_decision_sha256=identity,
            baseline_decision_sha256=identity,candidate_decision_sha256='f'*64,
            complete_original_decision_reconstructed=True,candidate_normalized_decision_exact=True,
            public_input_arrays_unchanged=True,execution_order=list(check.job.harness.original.execution_order(frame)),
            baseline_controller_s=.5,candidate_controller_s=.48))
    prior=({},dict(report=prior_report(),sensing_scope={'original_failure_frame':1173}),reference_rows)
    prior[0]['artifact_sha256']={'launch.json':'reference'}
    timing=check.job.check_rows(rows,reference_rows)
    launch=check.job.harness.scope() | dict(source_sha256={'fixed':'source'},
        reference_completion_sha256=check.job.COMPLETION_SHA,reference_result_sha256=check.job.RESULT_SHA,
        reference_artifact_sha256=prior[0]['artifact_sha256'],last_cpu_completion_sha256=check.job.LAST_CPU_SHA,
        component_benchmark_sha256=check.job.BENCHMARK_SHA,environment=check.job.previous.ENVIRONMENT,
        frames=1428,state_frames=list(check.job.harness.original.STATE_FRAMES),native_execution=False,
        model_training=False,automatic_retry=False)
    result=dict(status='DEFERRED_MEMO_SINGLE_PASS_LATE_HISTORY_V1_COMPLETE',source_sha256=launch['source_sha256'],
        artifact_sha256={'launch.json':check.LAUNCH_SHA,'comparison.jsonl':'comparisons'},
        report=check.job.expected_report(prior[1]['report'],timing),sensing_scope=prior[1]['sensing_scope'],
        original_inputs_reauthenticated_before_and_after=True,wall_s=1500.,native_execution=False,
        real_time_qualified=False,navigation_qualified=False,goal_achieved=False)
    return result,launch,prior,rows


def test_all_frames_and_negative_sensing_scope_are_preserved():
    values=payload();timing=check.check_payload(*values)
    assert timing['all_navigation']['candidate_total_s'] < timing['all_navigation']['baseline_total_s']
    assert values[0]['sensing_scope']=={'original_failure_frame':1173}


@pytest.mark.parametrize('fault',['short','baseline','input','original','timing','order','state','scope','source','wall','launch'])
def test_incomplete_or_changed_evidence_is_rejected(fault):
    result,launch,prior,rows=payload()
    if fault=='short': rows.pop()
    elif fault in ('baseline','input','original'):
        rows[1000][{'baseline':'baseline_decision_sha256','input':'public_input_sha256','original':'original_decision_sha256'}[fault]]='changed'
    elif fault=='timing': rows[1000]['candidate_controller_s']=-.1
    elif fault=='order': rows[1000]['execution_order']=[0,0]
    elif fault=='state': result['report']['observed_state_checks'][0]['state_sha256']='changed'
    elif fault=='scope': result['sensing_scope']={}
    elif fault=='source': result['source_sha256']={}
    elif fault=='wall': result['wall_s']=float('nan')
    else: launch['reference_result_sha256']='changed'
    with pytest.raises(ValueError): check.check_payload(result,launch,prior,rows)


def test_only_ended_original_owner_on_same_boot_is_accepted(monkeypatch):
    launch=dict(boot_id=check.job.previous.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),owner={})
    monkeypatch.setattr(check.job.previous,'owner_live',lambda _:True)
    with pytest.raises(ValueError,match='end first'): check.ended(launch)
    monkeypatch.setattr(check.job.previous,'owner_live',lambda _:False)
    check.ended(launch)
    launch['boot_id']='different'
    with pytest.raises(ValueError,match='boot'): check.ended(launch)
