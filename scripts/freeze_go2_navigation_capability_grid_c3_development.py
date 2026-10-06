"""Freeze the authorised indexing correction and exact exposure cutoffs."""
import hashlib
import json
import subprocess
from pathlib import Path
from lewm import decision_headroom_json_v42_development as output

REPO=Path(__file__).resolve().parents[1]


def binding(p):
    data=p.read_bytes()
    return dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def main():
    output.install(REPO/'docs')
    previous=REPO/'docs/go2_navigation_capability_correctness_c2_2026-09-26.json'
    p=json.loads(previous.read_text());base=Path(p['output_root'])
    result=json.loads((base/'cohorts/v0_startup_c2_C1_screen/result.json').read_text())
    assert result['complete'] and result['episodes']==10
    replay=json.loads((base/'grid_c3_bound_exposure/result.json').read_text())
    assert replay['status']=='PASS' and len(replay['rows'])==5
    evidence={}
    for i,row in zip((0,3,4,5,7),replay['rows'],strict=True):
        assert row['status']=='PASS' and row['native_arrays_exact_through_checked_prefix'] and row['consumed_sensor_hashes_bitwise']
        exposure=row['first_exposure']
        if exposure is None:assert row['full_source_checked']
        else:
            # A forecast outside a nominal envelope alone cannot loosen containment.
            assert any(e['kind'] in ('floor_observation','obstacle_observation','fine_obstacle_observation',
                'route_query','candidate_footprint') for e in exposure['events'])
        name=f'grid_c3_bound_exposure/dev{i:02d}/result.json'
        evidence[name]=binding(base/name)
    checks=json.loads((REPO/'docs/go2_navigation_capability_grid_c3_callsite_audit_2026-09-26.json').read_text())
    assert checks['after_expansion']['result']=='PASS'
    p['schema']='navigation_capability_correctness_c3.v1'
    p['predecessor']=dict(path=str(previous.relative_to(REPO)),**binding(previous))
    p['first_corrected_C0_assignment']='v0_grid_c3_gate_C0_dev00_ep0_attempt001'
    p['correction']=p['correction']|dict(harness='v0_grid_c3',outcome_iteration_versions_consumed=1,
        correctness_version_increment='0 only after refined containment passes',
        indexing='Shared resolutions, storage extents and array origins across routing, coverage, mapping and geometry; internal route centres use storage bound.',
        containment=dict(episodes=['00/0','03/0','04/0','05/0','07/0'],
            rule='Exact native arrays through exposure acquisition, consumed hashes inclusive through that frame, and requests before that acquisition. Whole recording exact if no exposure. Any prefix failure stops execution and further changes.',
            evidence=evidence,post_exposure_divergence='Report first divergence and explanation; no exact whole-recording requirement after actual old-bound exposure.'),
        tracker_estimation_unchanged=True,threshold_test='Full-domain red/green contract and source audit; see bound check report.')
    p['approval']='User guidance for next version, 26 September 2026: finish C2 unchanged as diagnostic only; C2 consumes one version; shared indexing correction, all-callsite/full-domain contract, refined first-exposure containment, stop before further changes on failure; pose-loss recovery diagnosis only after clean screen; estimator changes need approval.'
    protocol=REPO/'docs/go2_navigation_capability_correctness_c3_2026-09-26.json'
    with protocol.open('x') as f:json.dump(p,f,indent=2)
    predecessor=REPO/'docs/go2_navigation_capability_harness_v0_startup_c2_final_2026-09-26.json'
    h=json.loads(predecessor.read_text())
    h.update(version='v0_grid_c3',status='FROZEN_BEFORE_GRID_CORRECTION_SCREEN',
        predecessor_sha256=binding(predecessor)['sha256'],protocol_sha256=binding(protocol)['sha256'],
        correction=p['correction'],new_components=['Shared grid indexing correction','Refined exact-prefix containment'],
        correctness_only='Conditional on refined containment; stop on failure',
        outcome_driven_versions_consumed=1,refined_containment_evidence=evidence,
        controller_algorithms_changed=True,oracle_fidelity_check_pending=True)
    h.pop('preflight_supersedes',None)
    # Only reviewed current edits supersede old bindings; all others must match.
    changed=subprocess.check_output(['git','diff','HEAD','--name-only'],text=True).splitlines()
    added=[*changed,*checks['files']]
    names={x if isinstance(x,str) else x['path'] for x in added}
    names.update('scripts/'+x+'_development.py' for x in (
        'run_go2_navigation_capability_correctness_c3','read_go2_navigation_capability_correctness_c3',
        'read_go2_capability_grid_c3_failure','check_go2_capability_grid_c3_C0_replay',
        'project_go2_capability_grid_c3_budget','run_go2_capability_grid_c3_cohort',
        'replay_go2_capability_bound_exposure','run_go2_capability_bound_exposure_replays',
        'freeze_go2_navigation_capability_grid_c3'))
    names.update(('lewm/navigation_capability_refined_containment_development.py',
        'lewm/tests/test_navigation_capability_full_grid_contract_development.py',
        'lewm/tests/test_navigation_capability_refined_containment_development.py'))
    for name in names:
        assert all(not part.startswith('sealed') for part in Path(name).parts)
        if name.endswith('.py'):h['implementation_bindings'][name]=binding(REPO/name)
    h['implementation_bindings'][str(protocol.relative_to(REPO))]=binding(protocol)
    for name,bound in (h['original_source_bindings']|h['implementation_bindings']).items():
        assert binding(REPO/name)==bound, name
    target=REPO/'docs/go2_navigation_capability_harness_v0_grid_c3_final_2026-09-26.json'
    with target.open('x') as f:json.dump(h,f,indent=2)
    print(json.dumps(dict(protocol_sha256=binding(protocol)['sha256'],harness_sha256=binding(target)['sha256'])))


if __name__=='__main__':main()
