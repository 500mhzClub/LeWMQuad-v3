"""Freeze V3's single eligible-turn-memory change and amended gate sequence."""
import hashlib,json
from pathlib import Path
from lewm import decision_headroom_json_v42_development as output
REPO=Path(__file__).resolve().parents[1]
def binding(p):
    b=p.read_bytes();return dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
    output.install(REPO/'docs')
    previous=REPO/'docs/go2_navigation_capability_live_turn_v3_2026-09-27.json';p=json.loads(previous.read_text());base=Path(p['output_root'])
    result=json.loads((base/'cohorts/v2_exhausted_view_C1_screen/result.json').read_text());assert result['complete'] and result['successes']==8 and not result['passed']
    old=REPO/'docs/go2_navigation_capability_harness_v3_live_turn_final_2026-09-27.json';frozen=json.loads(old.read_text())
    for name,identity in (frozen['original_source_bindings']|frozen['implementation_bindings']).items():assert binding(REPO/name)==identity,name
    amendment=REPO/'docs/go2_navigation_capability_second_episode_gate_amendment_2026-09-27.json'
    p.update(schema='navigation_capability_live_turn_v3c1.v1',predecessor=dict(path=str(previous.relative_to(REPO)),**binding(previous)),
        approval='User 27 September: diagnose01/09, make one change, freeze and continue through amended C1 screens, C0, capability and videos in-session; stop only at authorised conditions.',
        first_corrected_C0_assignment='v3c1_live_turn_gate_C0_dev00_ep0_attempt001',gate_sequence_amendment=dict(path=str(amendment.relative_to(REPO)),**binding(amendment)))
    p['correction']=dict(harness='v3c1_live_turn',outcome_iteration_versions_consumed=4,outcome_version_increment=1,correctness_version_increment=0,
        single_change='Release visual turn-memory latch when its chosen direction becomes ineligible; then use unchanged ordinary selection and interruption memory.',
        rationale='01 retains blocked right-turn override for705 holds despite eligible left turn;09 predominantly has a different clearance/view oscillator.',
        expected_effect='Allow01 to leave the blocked visual-memory latch.09 may remain unchanged; no combined repair.',
        unchanged=['tracker estimation and pose admission','sensors and feature selection','models','candidate bank','mapping and routing cost','V2 reference exhaustion','clearance and stopping thresholds','dispatch guards','480-second mission budget','arrival rules'],
        predecessor_containment_passed=True,assignments=dict(screen=[[i,0]for i in range(10)],second_screen=[[i,1]for i in range(10)],gate=[[i,j]for i in range(10)for j in [0,1]]),
        gate_condition='Same-harness C1 first9/10 and second9/10 with zero safety violations, then C0>=19/20. Failed second check activates all20>=18/20 for future versions.',
        safety='Any disallowed contact or hard violation disqualifies the version and stops; preserve every failure, no retries.')
    p['implementation_erratum']=dict(reason='Deployed memory is InterruptedRouteTurnMemory, not SelectedRouteTurnMemory; replace the correct base with only the approved eligibility invalidation.',failed_attempt='v3_live_turn_screen_C1_dev00_ep0_attempt001',planning_decisions=0,policy_steps=0,outcome_version_increment=0,original_sources_preserved=True,refrozen_fresh_attempt=True)
    protocol=REPO/'docs/go2_navigation_capability_live_turn_v3c1_2026-09-27.json'
    with protocol.open('x')as f:json.dump(p,f,indent=2)
    frozen.update(version='v3c1_live_turn',status='FROZEN_BEFORE_SCREEN',predecessor_sha256=binding(old)['sha256'],protocol_sha256=binding(protocol)['sha256'],correction=p['correction'],new_components=['Live eligibility invalidates blocked visual route-turn latch'],correctness_only=False,outcome_driven_versions_consumed=4,oracle_fidelity_check_pending=True)
    names=['docs/go2_navigation_capability_live_turn_binding_erratum_2026-09-27.md','lewm/navigation_capability_live_turn_binding_c1_development.py','lewm/tests/test_navigation_capability_live_turn_binding_c1_development.py','scripts/diagnose_go2_capability_v2_turns_development.py','docs/go2_navigation_capability_v2_turn_diagnosis_2026-09-27.md',str(protocol.relative_to(REPO)),str(amendment.relative_to(REPO))]
    names.extend('scripts/'+n+'_development.py'for n in ['run_go2_navigation_capability_live_turn_v3c1','read_go2_navigation_capability_live_turn_v3c1','read_go2_capability_live_turn_v3c1_failure','check_go2_capability_live_turn_v3c1_C0_replay','project_go2_capability_live_turn_v3c1_budget','run_go2_capability_live_turn_v3c1_cohort','freeze_go2_navigation_capability_live_turn_v3c1'])
    for n in names:frozen['implementation_bindings'][n]=binding(REPO/n)
    frozen['diagnosis_evidence']={str(path.relative_to(base)):binding(path)for path in [base/'v2_turn_diagnosis_attempt001/result.json',base/'cohorts/v2_exhausted_view_C1_screen/result.json']}
    target=REPO/'docs/go2_navigation_capability_harness_v3c1_live_turn_final_2026-09-27.json'
    with target.open('x')as f:json.dump(frozen,f,indent=2)
    print(json.dumps(dict(protocol_sha256=binding(protocol)['sha256'],harness_sha256=binding(target)['sha256'])))
if __name__=='__main__':main()
