"""Freeze the two authorised startup corrections before their complete C1 screen."""
import hashlib
import json
from pathlib import Path
import subprocess
from lewm import decision_headroom_json_v42_development as output

REPO=Path(__file__).resolve().parents[1]


def binding(p):
    data=p.read_bytes();return dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def main():
    output.install(REPO/'docs')
    previous=REPO/'docs/go2_navigation_capability_correctness_c1_2026-09-26.json'
    p=json.loads(previous.read_text());base=Path(p['output_root'])
    old=json.loads((base/'cohorts/v0_task_c1_C1_screen/result.json').read_text());assert old['complete']
    diag=json.loads((base/'initial_floor_01_first_frame_diagnosis/result.json').read_text());assert diag['all_consumed_frame_hash_fields_match']
    p['schema']='navigation_capability_correctness_c2.v1'
    p['predecessor']=dict(path=str(previous.relative_to(REPO)),**binding(previous))
    p['first_corrected_C0_assignment']='v0_startup_c2_gate_C0_dev00_ep0_attempt001'
    p['correction']=dict(harness='v0_startup_c2',outcome_iteration_versions_consumed='0 provisional; 1 if containment fails',
        task_reference='Unchanged approved one-time settled transform for both cues',
        map_domain='Generator maximum 5.5x5.5 m envelope: ceil(diagonal+0.1)=8 m half width; 7.9-m point guard',
        startup='Primary floor unchanged when >=100 quads; otherwise existing auxiliary >=100. If both insufficient, retain frame-zero origin, retry acquisitions, wait 1 s then gyro-tracked guarded left turn. Stop at 24 s or 2pi cumulative gyro travel. All time counts against 480 s.',
        containment=dict(episodes=['00/0','03/0','04/0','05/0','07/0'],
            rule='All old episodes that reached planning, including potential extent exposure: identical native arrays, commands and consumed sensor hashes; any divergence consumes a harness version and must be diagnosed.'),
        minimum_floor_quads_unchanged=100,controller_models_unchanged=True,sensors_unchanged=True,candidate_bank_unchanged=True,
        structural_counts_path='startup_geometry_correctness_c2_2026-09-26/result.json',
        measured_first_frame_diagnosis='initial_floor_01_first_frame_diagnosis/result.json',
        threshold_test='Map-domain contract failed before fix (allocation); passed after fix.')
    p['approval']='User adjustment to post-screen plan, 26 September 2026: leave screen unchanged; aggregate structural checks all180; combined frozen map/startup correctness fix; rerun all10; exact containment for unaffected episodes else count version; then ordinary one-change iteration.'
    path=REPO/'docs/go2_navigation_capability_correctness_c2_2026-09-26.json'
    with path.open('x') as f:json.dump(p,f,indent=2)
    predecessor=REPO/'docs/go2_navigation_capability_harness_v0_task_c1_2026-09-26.json'
    h=json.loads(predecessor.read_text())
    h.update(version='v0_startup_c2',status='FROZEN_BEFORE_STARTUP_CORRECTION_SCREEN',
        predecessor_sha256=binding(predecessor)['sha256'],correction=p['correction'],
        controller_algorithms_changed=True,correctness_only='Conditional on exact containment',
        outcome_driven_versions_consumed='Pending containment: 0 or 1',
        new_components=['Generator-sized map domain','Shared measured initial-floor fallback and bounded recovery'],
        oracle_fidelity_check_pending=True)
    # Preserve predecessor document. Bind each current source version explicitly.
    for group in ('original_source_bindings','implementation_bindings'):
        h[group]={name:binding(REPO/name) for name in h[group]}
    added=[*subprocess.check_output(['git','diff','--name-only'],text=True).splitlines(),
        'lewm/navigation_capability_map_domain_development.py',
        'lewm/navigation_capability_startup_recovery_development.py',
        'lewm/navigation_capability_startup_containment_development.py',
        'lewm/tests/test_navigation_capability_startup_recovery_development.py',
        'scripts/check_go2_capability_startup_geometry_development.py',
        'scripts/check_go2_capability_map_domain_contract_development.py',
        'scripts/diagnose_go2_capability_initial_floor_development.py',
        'scripts/run_go2_navigation_capability_correctness_c2_development.py',
        'scripts/read_go2_navigation_capability_correctness_c2_development.py',
        'scripts/read_go2_capability_startup_c2_failure_development.py',
        'scripts/run_go2_capability_startup_c2_cohort_development.py',
        'scripts/project_go2_capability_startup_c2_budget_development.py',
        'scripts/check_go2_capability_startup_c2_C0_replay_development.py',
        'scripts/freeze_go2_navigation_capability_startup_c2_development.py',
        str(path.relative_to(REPO))]
    for name in added:
        if name.endswith('.py') or name==str(path.relative_to(REPO)):h['implementation_bindings'][name]=binding(REPO/name)
    with (REPO/'docs/go2_navigation_capability_harness_v0_startup_c2_2026-09-26.json').open('x') as f:json.dump(h,f,indent=2)
    print('Frozen startup correction configuration and source bindings')

if __name__=='__main__':main()
