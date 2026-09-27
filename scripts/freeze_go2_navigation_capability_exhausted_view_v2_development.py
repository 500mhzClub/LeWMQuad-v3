"""Freeze the single shared exhausted-view recovery change and unchanged screen."""
import hashlib
import json
from pathlib import Path
from lewm import decision_headroom_json_v42_development as output
REPO=Path(__file__).resolve().parents[1]
def binding(p):
    b=p.read_bytes();return dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def main():
    output.install(REPO/'docs')
    predecessor=REPO/'docs/go2_navigation_capability_paired_floor_v1_2026-09-26.json'
    p=json.loads(predecessor.read_text());base=Path(p['output_root'])
    screen=json.loads((base/'cohorts/v1_paired_floor_C1_screen/result.json').read_text())
    assert screen['complete'] and screen['episodes']==10 and not screen['passed']
    contract=json.loads((base/'paired_floor_output_contract_attempt001/result.json').read_text())
    assert contract['status']=='PASS'
    old=REPO/'docs/go2_navigation_capability_harness_v1_paired_floor_final_2026-09-26.json'
    frozen=json.loads(old.read_text())
    for name,identity in (frozen['original_source_bindings']|frozen['implementation_bindings']).items():assert binding(REPO/name)==identity,name
    p.update(schema='navigation_capability_exhausted_view_v2.v1',
        predecessor=dict(path=str(predecessor.relative_to(REPO)),**binding(predecessor)),
        approval='Approved brief sections 4–9; user 27 September: leg-split timeouts/pose loss, inspect turnaround and outbound-route reuse, flag budget pressure without changing budget, then choose the next change from evidence.',
        first_corrected_C0_assignment='v2_exhausted_view_gate_C0_dev00_ep0_attempt001')
    p['correction']=dict(harness='v2_exhausted_view',outcome_iteration_versions_consumed=3,outcome_version_increment=1,correctness_version_increment=0,
        single_change='Retire and consume an attained measured-view recovery reference after one second of continuous accepted-pose alignment without inherited strong-support release.',
        alignment_tolerance_rad=.1,dwell_ns=1_000_000_000,maximum_observation_gap_ns=100_000_000,
        rationale='05 return and 08 outbound spend 306.4/318.8 seconds holding; 569/578 recovery holds are already aligned within 0.1 rad with a clearance-eligible turn.',
        expected_effect='End futile attained-reference holding and permit ordinary routing; no claim that other recorded turn-memory or pose-loss mechanisms are fixed.',
        unchanged=['tracker estimation and pose admission','sensors and feature selection','models','candidate bank','mapping','routing cost','clearance and stopping rules','dispatch guards','480-second mission budget','arrival rules'],
        predecessor_containment_passed=True,assignments=dict(screen=[[i,0] for i in range(10)],gate=[[i,j] for i in range(10) for j in (0,1)]),
        gate_condition='C1 >=9/10, zero contacts/hard violations and all sampled hard clearance qualified; then unchanged C0 >=19/20 gate.',
        safety='Any disallowed contact or hard violation disqualifies this version; preserve failures, no automatic retries.')
    protocol=REPO/'docs/go2_navigation_capability_exhausted_view_v2_2026-09-27.json'
    with protocol.open('x') as f:json.dump(p,f,indent=2)
    frozen.update(version='v2_exhausted_view',status='FROZEN_BEFORE_SCREEN',predecessor_sha256=binding(old)['sha256'],
        protocol_sha256=binding(protocol)['sha256'],correction=p['correction'],new_components=['Exhaustible measured-view recovery reference'],
        correctness_only=False,outcome_driven_versions_consumed=3,oracle_fidelity_check_pending=True)
    names=['lewm/navigation_capability_exhausted_view_development.py','lewm/tests/test_navigation_capability_exhausted_view_development.py',
        'scripts/analyse_go2_capability_paired_floor_legs_development.py','docs/go2_navigation_capability_paired_floor_leg_diagnosis_2026-09-27.md',
        'docs/go2_navigation_capability_paired_floor_leg_diagnosis_2026-09-27.json',str(protocol.relative_to(REPO))]
    names.extend('scripts/'+s+'_development.py' for s in ['run_go2_navigation_capability_exhausted_view_v2','read_go2_navigation_capability_exhausted_view_v2',
        'read_go2_capability_exhausted_view_v2_failure','check_go2_capability_exhausted_view_v2_C0_replay','project_go2_capability_exhausted_view_v2_budget',
        'run_go2_capability_exhausted_view_v2_cohort','freeze_go2_navigation_capability_exhausted_view_v2'])
    for name in names:frozen['implementation_bindings'][name]=binding(REPO/name)
    frozen['diagnosis_evidence']={str(path.relative_to(base)):binding(path) for path in [base/'cohorts/v1_paired_floor_C1_screen/result.json',
        base/'paired_floor_output_contract_attempt001/result.json',base/'paired_floor_leg_diagnosis_attempt001/result.json',base/'paired_floor_leg_diagnosis_attempt001/mechanism_witnesses.json']}
    target=REPO/'docs/go2_navigation_capability_harness_v2_exhausted_view_final_2026-09-27.json'
    with target.open('x') as f:json.dump(frozen,f,indent=2)
    print(json.dumps(dict(protocol_sha256=binding(protocol)['sha256'],harness_sha256=binding(target)['sha256'])))
if __name__=='__main__':main()
