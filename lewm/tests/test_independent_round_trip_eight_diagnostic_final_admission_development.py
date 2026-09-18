from copy import deepcopy
import json
import pytest

from scripts import independent_round_trip_eight_diagnostic_final_admission_development as final


@pytest.fixture(scope='module')
def fixed_manifest():
    return final.inputs.study.manifest()


@pytest.fixture(autouse=True)
def unchanged_manifest(monkeypatch,fixed_manifest):
    # Construct the actual deterministic inventory once; keep each test's
    # mutable review isolated without regenerating the same eight layouts.
    monkeypatch.setattr(final.inputs.study,'manifest',lambda:deepcopy(fixed_manifest))


def admissions():
    batch='a'*64; ids={k:str(i)*64 for i,k in enumerate(('frontier','hold','contact'),1)}
    first=dict(adapter_batch_result_sha256=batch,ordered_waiter_result_sha256=ids,
        completed=[dict(stage=k,waiter_result_sha256=h,native_result_sha256='b'*64,
            case=k,collection={'terminal':'failed'},readout={'verified_round_trip':False},
            measured_round_trip_successes=0) for k,h in ids.items()])
    four=dict(original_queue_admission=first,tracking_wait_result_sha256='4'*64,
              tracking_completion={'native_result_sha256':'c'*64})
    five=dict(original_four_stage_admission=four,budget_wait_result_sha256='5'*64,
        budget_completion={'native_result_sha256':'d'*64},budget_inputs=dict(batch=batch,**ids,tracking='4'*64))
    inputs=dict(adapter_batch_result_sha256=batch,development_summary={
        'measured_round_trip_successes':0,'outcomes':[{'verified_round_trip':False} for _ in range(6)]})
    return inputs,dict(original_five_stage_admission=five,all_eight_diagnostics_authenticated=True,
        later_waiter_result_sha256={s.name:str(i+6)*64 for i,s in enumerate(final.queue.STAGES)},
        later_diagnostics=[dict(stage=s.name,readout={'verified_round_trip':False},
            prefix_comparison={'exact':False},measured_round_trip_successes=0) for s in final.queue.STAGES])


def evidence(monkeypatch,inputs,queue):
    def diagnostic(native,completion):
        return dict(native_result_sha256=completion['native_result_sha256'],case='synthetic',
            collection={'terminal':'failed'},readout={'verified_round_trip':False},
            prefix_comparison={'budget_only_preboundary_execution_supported':False})
    monkeypatch.setattr(final.original,'diagnostic_readout',diagnostic)
    return final.review_evidence(inputs,queue)


def review(inputs,queue,proof):
    return dict(schema='independent_round_trip_eight_diagnostic_final_policy_review.v1',
        status='COMPLETE_EIGHT_DIAGNOSTIC_POLICY_REVIEW',
        decision='execute_original_fixed_32_case_definition',
        study_manifest=final.inputs.study.manifest(),navigation_ticks=final.NAVIGATION_TICKS,
        input_admission_sha256=final.fingerprint(inputs),eight_stage_queue_admission_sha256=final.fingerprint(queue),
        development_evidence=deepcopy(proof),all_scientific_failures_retained=True,
        policy_or_budget_changes_requested=False,independent_layout_sensor_data_consumed=False,
        prior_navigation_qualification_claimed=False,prior_real_time_qualification_claimed=False,
        prior_hardware_qualification_claimed=False,rationale={k:'Synthetic review rationale for '+k for k in final.REASONS})


def test_all_negative_results_remain_valid_review_evidence(monkeypatch):
    inp,q=admissions();before=deepcopy((inp,q));proof=evidence(monkeypatch,inp,q)
    final.require_review(review(inp,q,proof),inp,q,proof)
    assert len(proof['original_five_diagnostic_evidence']['original_three_diagnostics'])==3
    assert proof['original_five_diagnostic_evidence']['tracking']['readout']['verified_round_trip'] is False
    assert proof['original_five_diagnostic_evidence']['extended_budget']['prefix_comparison']['budget_only_preboundary_execution_supported'] is False
    assert (inp,q)==before


@pytest.mark.parametrize('fault',['batch','stage','tracking','budget_inputs'])
def test_mixed_or_incomplete_diagnostic_chain_rejected(monkeypatch,fault):
    inp,q=admissions()
    if fault=='batch':inp['adapter_batch_result_sha256']='f'*64
    elif fault=='stage':q['original_five_stage_admission']['original_four_stage_admission']['original_queue_admission']['completed'].pop()
    elif fault=='tracking':q['original_five_stage_admission']['original_four_stage_admission']['tracking_wait_result_sha256']='f'*64
    else:q['original_five_stage_admission']['budget_inputs'].pop('hold')
    with pytest.raises(ValueError):evidence(monkeypatch,inp,q)


@pytest.mark.parametrize('fault',['decision','budget','model','order','input','queue','outcome',
    'prefix','scope','rationale','missing_reason','missing_field','extra_field','typed_flag'])
def test_review_cannot_substitute_policy_outcomes_or_qualification(monkeypatch,fault):
    inp,q=admissions();proof=evidence(monkeypatch,inp,q);r=review(inp,q,proof)
    if fault=='decision':r['decision']='needs_revision'
    elif fault=='budget':r['navigation_ticks']=4000
    elif fault=='model':r['study_manifest']['ordered_cases'][0]['assignment']['model_state_sha256']='f'*64
    elif fault=='order':r['study_manifest']['ordered_cases'].reverse()
    elif fault=='input':r['input_admission_sha256']='f'*64
    elif fault=='queue':r['eight_stage_queue_admission_sha256']='f'*64
    elif fault=='outcome':r['development_evidence']['original_five_diagnostic_evidence']['six_case_development_summary']['measured_round_trip_successes']=1
    elif fault=='prefix':r['development_evidence']['original_five_diagnostic_evidence']['extended_budget']['prefix_comparison']['budget_only_preboundary_execution_supported']=True
    elif fault=='scope':r['prior_real_time_qualification_claimed']=True
    elif fault=='rationale':r['rationale']['navigation_budget']=' '
    elif fault=='missing_reason':r['rationale'].pop('sensing_failures')
    elif fault=='missing_field':r.pop('development_evidence')
    elif fault=='extra_field':r['permit_new_policy']=True
    else:r['all_scientific_failures_retained']=1
    with pytest.raises(ValueError):final.require_review(r,inp,q,proof)


def test_full_queue_reconstruction_chains_original_five_and_all_later_outputs(monkeypatch):
    _,q=admissions();calls=[];five={'authenticated_five':'receipt'}
    def original(value,sources):
        assert value is q['original_five_stage_admission'];calls.append('five');return five
    def later(value,ids,*,sources):
        assert value is five and ids is q['later_waiter_result_sha256'];calls.append('later');return q
    monkeypatch.setattr(final.original,'reconstruct_full_queue',original)
    monkeypatch.setattr(final.queue,'admit',later)
    assert final.reconstruct_full_queue(q,{}) is q
    assert calls==['five','later']


def test_live_owner_stops_before_sources_inputs_or_review(monkeypatch):
    def live():raise ValueError('original diagnostic still live')
    monkeypatch.setattr(final.queue,'owners_ended',live)
    monkeypatch.setattr(final,'prepared_sources',lambda:pytest.fail('source work after live rejection'))
    with pytest.raises(ValueError,match='still live'):final.verify_population({})


def prepared_launch(monkeypatch,tmp_path):
    inp,q=admissions();proof=evidence(monkeypatch,inp,q);r=review(inp,q,proof)
    path=tmp_path/final.REVIEW;path.parent.mkdir();path.write_text(json.dumps(r))
    from scripts.run_go2_successive_choice_maze_development_v1 import digest,ROOT
    sources={final.SOURCE:digest(ROOT/final.SOURCE),final.REVIEW:digest(path)}
    launch=dict(source_sha256=sources,runtime_verifier=dict(source=final.SOURCE,function='verify_population'),
        policy_review=dict(path=final.REVIEW,sha256=sources[final.REVIEW]),input_admission=inp,
        eight_stage_queue_admission=q,final_policy_review_completed=True,
        complete_input_admission_performed=True,native_queue_completion_verified=True,all_eight_diagnostics_reviewed=True,
        ordered_cases=final.inputs.study.manifest()['ordered_cases'],runtime=deepcopy(final.runtime.FIXED_RUNTIME),
        staged_runtime=deepcopy(final.overlap.driver.staged.FIXED),audit_cpu_monitor=deepcopy(final.overlap.monitored.FIXED),
        overlap_evidence=deepcopy(final.overlap.EVIDENCE),population_entrypoint=deepcopy(final.overlap.ENTRYPOINT),
        overlap_verifier=dict(source=final.overlap.SOURCE,function='verify_overlap'))
    calls=[]
    monkeypatch.setattr(final,'ROOT',tmp_path)
    monkeypatch.setattr(final,'prepared_sources',lambda:{final.SOURCE:sources[final.SOURCE]})
    def verify(selected):
        for n,h in selected.items():assert digest((ROOT if n==final.SOURCE else tmp_path)/n)==h
    monkeypatch.setattr(final,'verify',verify)
    monkeypatch.setattr(final.queue,'owners_ended',lambda:calls.append('owners'))
    monkeypatch.setattr(final.inputs,'verify_bound',lambda *a:calls.append('inputs'))
    monkeypatch.setattr(final.queue,'verify_bound',lambda *a:calls.append('queue'))
    monkeypatch.setattr(final.inputs,'admit',lambda *a:(calls.append('full_inputs') or deepcopy(inp)))
    monkeypatch.setattr(final,'reconstruct_full_queue',lambda *a:(calls.append('full_queue') or deepcopy(q)))
    return launch,calls


@pytest.mark.parametrize('full',[False,True])
def test_joined_verifier_calls_evidence_verifiers_and_preserves_launch(monkeypatch,tmp_path,full):
    launch,calls=prepared_launch(monkeypatch,tmp_path);before=deepcopy(launch)
    assert final.verify_population(launch,full=full) is None
    assert launch==before
    assert calls==['owners','inputs','queue']+(['full_inputs','full_queue'] if full else [])+['owners']


@pytest.mark.parametrize('fault',['input_rejected','queue_rejected','full_input_changed','full_queue_changed',
                                 'monitor','driver','review_path','review_binding','runtime_case'])
def test_launch_flags_do_not_bypass_actual_checks(monkeypatch,tmp_path,fault):
    launch,_=prepared_launch(monkeypatch,tmp_path)
    def reject(*a):raise ValueError('actual evidence rejected')
    if fault=='input_rejected':monkeypatch.setattr(final.inputs,'verify_bound',reject)
    elif fault=='queue_rejected':monkeypatch.setattr(final.queue,'verify_bound',reject)
    elif fault=='full_input_changed':monkeypatch.setattr(final.inputs,'admit',lambda *a:{})
    elif fault=='full_queue_changed':monkeypatch.setattr(final,'reconstruct_full_queue',lambda *a:{})
    elif fault=='monitor':launch['audit_cpu_monitor']={}
    elif fault=='driver':launch['population_entrypoint']['function']='unmonitored'
    elif fault=='review_path':launch['policy_review']['path']='unexpected.json'
    elif fault=='review_binding':launch['policy_review']['sha256']='f'*64
    else:launch['ordered_cases'].reverse()
    with pytest.raises(ValueError):final.verify_population(launch,full=True)


@pytest.mark.parametrize('fault',['later_outcome','later_missing','later_link','later_reason','old_schema'])
def test_review_cannot_omit_or_rewrite_later_diagnostics(monkeypatch,fault):
    inp,q=admissions();proof=evidence(monkeypatch,inp,q);r=review(inp,q,proof)
    if fault=='later_outcome':r['development_evidence']['later_diagnostics'][1]['measured_round_trip_successes']=1
    if fault=='later_missing':r['development_evidence']['later_diagnostics'].pop()
    if fault=='later_link':r['eight_stage_queue_admission_sha256']='0'*64
    if fault=='later_reason':r['rationale'].pop('later_diagnostic_outcomes')
    if fault=='old_schema':r['schema']='independent_round_trip_final_policy_review.v1'
    with pytest.raises(ValueError):final.require_review(r,inp,q,proof)
