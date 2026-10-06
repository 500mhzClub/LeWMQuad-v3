"""Freeze the fifth version: recovery uses existing actual tracker features."""
import hashlib,json
from pathlib import Path
from lewm import decision_headroom_json_v42_development as output
REPO=Path(__file__).resolve().parents[1]
def binding(p):
    b=p.read_bytes();return dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
    output.install(REPO/'docs');previous=REPO/'docs/go2_navigation_capability_live_turn_v3c1_2026-09-27.json';p=json.loads(previous.read_text());base=Path(p['output_root'])
    r=json.loads((base/'cohorts/v3c1_live_turn_C1_screen/result.json').read_text());assert r['complete'] and r['successes']==8 and not r['passed']
    assert json.loads((base/'v3_support_replay_attempt001/result.json').read_text())['status']=='PASS'
    old=REPO/'docs/go2_navigation_capability_harness_v3c1_live_turn_final_2026-09-27.json';frozen=json.loads(old.read_text())
    for n,b in (frozen['original_source_bindings']|frozen['implementation_bindings']).items():assert binding(REPO/n)==b,n
    p.update(schema='navigation_capability_completed_support_v4.v1',predecessor=dict(path=str(previous.relative_to(REPO)),**binding(previous)),first_corrected_C0_assignment='v4_completed_support_gate_C0_dev00_ep0_attempt001')
    p['predecessor_implementation_erratum']=p.pop('implementation_erratum')
    p['correction']=dict(harness='v4_completed_support',outcome_iteration_versions_consumed=5,outcome_version_increment=1,correctness_version_increment=0,
        single_change='Existing recovery uses actual selected tracker feature counts including existing sparse corner completion, instead of only original strong-corner subset.',
        rationale='Exact replay:197 frames across inspected01/09 windows have original strong maximum<48 but actual selected maximum>=85, while poses remain admitted.',
        expected_effect='Avoid needless view-recovery reversals; risk of missing early tracking weakness remains to be tested.',
        unchanged=['feature extraction and selection','tracker estimation and admission','sensors','models','candidate bank','48/96 recovery thresholds','V2 reference exhaustion','V3 blocked-latch release','routing and mapping','safety and dispatch gates','480-second budget','arrival rules'],
        predecessor_containment_passed=True,assignments=dict(screen=[[i,0]for i in range(10)],second_screen=[[i,1]for i in range(10)],gate=[[i,j]for i in range(10)for j in [0,1]]),
        gate_condition='Same-harness C1 first9/10 and second9/10 with zero safety violations, then C0>=19/20. Failed second check activates all20>=18/20 for future versions.',
        safety='Disallowed contact or hard violation stops and disqualifies this version. No silent retries.')
    protocol=REPO/'docs/go2_navigation_capability_completed_support_v4_2026-09-27.json'
    with protocol.open('x')as f:json.dump(p,f,indent=2)
    frozen.update(version='v4_completed_support',status='FROZEN_BEFORE_SCREEN',predecessor_sha256=binding(old)['sha256'],protocol_sha256=binding(protocol)['sha256'],correction=p['correction'],new_components=['Actual selected tracker-feature recovery count'],correctness_only=False,outcome_driven_versions_consumed=5,oracle_fidelity_check_pending=True)
    names=['lewm/navigation_capability_completed_support_development.py','lewm/tests/test_navigation_capability_completed_support_development.py','docs/go2_navigation_capability_v3_support_diagnosis_2026-09-27.md','docs/go2_navigation_capability_v3_support_replay_plan_2026-09-27.json','scripts/replay_go2_capability_v3_recovery_support_development.py',str(protocol.relative_to(REPO))]
    names+=['scripts/'+s+'_development.py'for s in ['run_go2_navigation_capability_completed_support_v4','read_go2_navigation_capability_completed_support_v4','read_go2_capability_completed_support_v4_failure','check_go2_capability_completed_support_v4_C0_replay','project_go2_capability_completed_support_v4_budget','run_go2_capability_completed_support_v4_cohort','freeze_go2_navigation_capability_completed_support_v4']]
    for n in names:frozen['implementation_bindings'][n]=binding(REPO/n)
    frozen['diagnosis_evidence']={str(q.relative_to(base)):binding(q)for q in [base/'cohorts/v3c1_live_turn_C1_screen/result.json',base/'v3_support_replay_attempt001/result.json',base/'v3_support_replay_attempt001/dev01/result.json',base/'v3_support_replay_attempt001/dev09/result.json']}
    target=REPO/'docs/go2_navigation_capability_harness_v4_completed_support_final_2026-09-27.json'
    with target.open('x')as f:json.dump(frozen,f,indent=2)
    print(json.dumps(dict(protocol_sha256=binding(protocol)['sha256'],harness_sha256=binding(target)['sha256'])))
if __name__=='__main__':main()
