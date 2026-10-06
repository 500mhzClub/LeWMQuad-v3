"""Freeze one outcome-driven mapping change after the complete failure diagnosis."""
import hashlib
import json
from pathlib import Path
from lewm import decision_headroom_json_v42_development as output

REPO=Path(__file__).resolve().parents[1]


def binding(path):
    data=path.read_bytes()
    return dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def main():
    output.install(REPO/'docs')
    predecessor=REPO/'docs/go2_navigation_capability_correctness_c3_2026-09-26.json'
    protocol=json.loads(predecessor.read_text());base=Path(protocol['output_root'])
    screen=json.loads((base/'cohorts/v0_grid_c3_C1_screen/result.json').read_text())
    assert screen['complete'] and screen['refined_containment_passed'] and screen['episodes']==10
    for name in ('grid_c3_pose_loss_diagnosis_attempt001','grid_c3_initial_map_diagnosis_attempt001'):
        assert json.loads((base/name/'result.json').read_text())['status']=='PASS'
    protocol.update(schema='navigation_capability_paired_floor_v1.v1',
        predecessor=dict(path=str(predecessor.relative_to(REPO)),**binding(predecessor)),
        approval='Navigation brief sections 4–9 and latest instruction: diagnose four timeouts alongside three pose losses, check shared startup cause, choose one evidence-based shared change; safety unchanged.',
        first_corrected_C0_assignment='v1_paired_floor_gate_C0_dev00_ep0_attempt001')
    protocol['correction']=dict(harness='v1_paired_floor',outcome_iteration_versions_consumed=2,
        outcome_version_increment=1,correctness_version_increment=0,
        single_change='Initial fixed scalar map floor from existing qualified current paired depth plane, replacing unqualified mesh-candidate median; retain original map orientation and bounded startup recovery.',
        rationale='All five auxiliary-start failures initialise map floor 0.234–0.290 m above the paired measured plane; the primary-start control differs by <1 mm.',
        expected_effect='Restore measured floor and obstacle-height interpretation, allowing observed routes; may prevent associated unnecessary wall-facing scans. Navigation success remains unproven.',
        unchanged=['tracker estimation','plane acceptance thresholds','sensor packets and cameras','models','candidate bank','subsequent map update','routing and recovery policies','all safety/clearance/dispatch rules'],
        predecessor_containment_passed=True,
        assignments=dict(screen=[[i,0] for i in range(10)],gate=[[i,j] for i in range(10) for j in (0,1)]),
        gate_condition='C1 >=9/10, zero contacts/hard violations and all sampled hard clearance qualified; then unchanged C0 >=19/20 gate.',
        safety='Any disallowed contact or hard violation disqualifies this version; preserve every failure, no automatic retries.')
    protocol_path=REPO/'docs/go2_navigation_capability_paired_floor_v1_2026-09-26.json'
    with protocol_path.open('x') as stream:json.dump(protocol,stream,indent=2)
    old=REPO/'docs/go2_navigation_capability_harness_v0_grid_c3_final_2026-09-26.json'
    frozen=json.loads(old.read_text())
    for name,identity in (frozen['original_source_bindings']|frozen['implementation_bindings']).items():
        assert binding(REPO/name)==identity,name
    frozen.update(version='v1_paired_floor',status='FROZEN_BEFORE_FIRST_TUNING_SCREEN',
        predecessor_sha256=binding(old)['sha256'],protocol_sha256=binding(protocol_path)['sha256'],
        correction=protocol['correction'],new_components=['Qualified paired-plane initial scalar map floor'],
        correctness_only=False,outcome_driven_versions_consumed=2,oracle_fidelity_check_pending=True)
    names=['lewm/navigation_capability_paired_floor_start_development.py',
           'lewm/tests/test_navigation_capability_paired_floor_start_development.py',str(protocol_path.relative_to(REPO)),
           'docs/go2_navigation_capability_grid_c3_failure_diagnosis_2026-09-26.md']
    names.extend('scripts/'+s+'_development.py' for s in (
        'run_go2_navigation_capability_paired_floor_v1','read_go2_navigation_capability_paired_floor_v1',
        'read_go2_capability_paired_floor_v1_failure','check_go2_capability_paired_floor_v1_C0_replay',
        'project_go2_capability_paired_floor_v1_budget','run_go2_capability_paired_floor_v1_cohort',
        'freeze_go2_navigation_capability_paired_floor_v1','analyse_go2_capability_grid_c3_failures_readonly',
        'diagnose_go2_capability_grid_c3_initial_map'))
    for name in names:frozen['implementation_bindings'][name]=binding(REPO/name)
    frozen['diagnosis_evidence']={str(path.relative_to(base)):binding(path) for path in (
        base/'cohorts/v0_grid_c3_C1_screen/result.json',
        base/'grid_c3_failure_log_diagnosis_attempt001/result.json',
        base/'grid_c3_pose_loss_diagnosis_attempt001/result.json',
        base/'grid_c3_initial_map_diagnosis_attempt001/result.json')}
    target=REPO/'docs/go2_navigation_capability_harness_v1_paired_floor_final_2026-09-26.json'
    with target.open('x') as stream:json.dump(frozen,stream,indent=2)
    print(json.dumps(dict(protocol_sha256=binding(protocol_path)['sha256'],harness_sha256=binding(target)['sha256'])))


if __name__=='__main__':main()
