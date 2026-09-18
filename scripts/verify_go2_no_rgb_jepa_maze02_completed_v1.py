"""Authenticate completed no-RGB JEPA evidence and reconstruct native traversal."""
from collections import Counter
import datetime
import json
from pathlib import Path

import numpy as np

from scripts import run_go2_all_phase_adapter_maze02_matched_native_v1 as batch
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json, read_npz
from scripts.maze_decision_stream_development import read_rows
from lewm.novel_maze_round_trip_evaluation_development import evaluate

SOURCE = 'scripts/verify_go2_no_rgb_jepa_maze02_completed_v1.py'
OUTPUT = Path('docs/go2_all_phase_adapter_no_rgb_jepa_maze02_verification_2026-09-10.json')
CASE = 'all_phase_no_rgb_jepa_residual_maze_02'
LAUNCH_SHA = '97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a'
WORKER_SHA = '5bc0c40bda4435bfaf6e6616e81981d23b1d6616f0530435b1a5c74dba4dad60'
PARENT_SHA = 'd3597bc5834afbafbf9e9bb56b8c35fa92346d2ddd98cbb2a21fc04d2224d93a'


def main():
    if not __debug__: raise ValueError('assertions required')
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completed-evidence verification')
    root = batch.OUTPUT
    anchors = {'launch.json':LAUNCH_SHA, CASE+'_worker_terminal.json':WORKER_SHA,
               CASE+'_parent_completion.json':PARENT_SHA}
    verify_artifacts(root,anchors)
    launch = read_json(root,'launch.json'); worker = read_json(root,CASE+'_worker_terminal.json')
    sources = discover_sources((SOURCE,),launch['source_sha256']); verify(sources)
    ids = anchors | worker['artifact_sha256'] | {CASE+'_worker.log':worker['worker_log_sha256']}
    verify_artifacts(root,ids)
    case = next(c for c in batch.CASES if c[0] == CASE)
    assert case == batch.CASES[3]
    audit = read_json(root,CASE+'_audit.json'); collection = worker['collection']; directory = root/CASE
    batch.require_case(case,worker,audit)
    assert read_json(directory,'result.json') == collection
    expected = {CASE+'/'+n for n in batch.artifacts(case[1],collection)}
    expected |= {CASE+s for s in ('_audit.json','_startup_comparison.json','_readout.json')}
    assert set(worker['artifact_sha256']) == expected
    assert worker['model_state_sha256'] == launch['assigned_model_states'][case[4]]
    parent = read_json(root,CASE+'_parent_completion.json')
    assert parent == dict(case=CASE,worker_terminal_sha256=WORKER_SHA,
                          verified_round_trip=worker['verified_round_trip'],scientific_success_required=False)
    startup = batch.compare_adapter_startup(case,directory)
    assert startup == worker['startup_comparison'] == read_json(root,CASE+'_startup_comparison.json')
    print('NO_RGB_JEPA_ARTIFACTS_AND_STARTUP_AUTHENTICATED',len(ids),flush=True)
    actions = Counter(); terminal = None; count = 0; final_mission = None
    for row in read_rows(directory):
        assert row['tick'] == row['observation_index'] == count
        decision = row['decision']; final_mission = decision['mission_receipt']
        if decision['new_selection'] is not None:
            actions[str(decision['new_selection']['action'])] += 1
        if terminal is None and decision['terminal'] is not None:
            terminal = dict(frame=count,terminal=decision['terminal'],failure=decision['failure'],
                requested_command=decision['requested_command'],
                mission_receipt=decision['mission_receipt'],
                evidence=decision.get('evidence'))
        count += 1
    assert count == collection['decisions'] == collection['rgbd_frames']
    assert dict(actions) == audit['selected_actions'] and final_mission == collection['mission_receipt']
    raw = read_npz(directory,'physics_trace.npz')
    native = evaluate(raw,final_mission or dict(arrivals=[],terminal=None),collection,layout_index=case[1])
    assert native == audit['native_evaluation'] == worker['native_evaluation']
    readout = batch.case_readout(audit,collection,raw['physics_contact'])
    assert readout == worker['readout'] == read_json(root,CASE+'_readout.json')
    assert audit['strict_physical_visibility_pass'] is True and audit['hard_measurement_failed_frames'] == []
    assert all(r['physical_visibility']['passes_sampled_physical_visibility'] for r in audit['depth_checks'])
    assert all(r['auxiliary_visibility_pass'] for r in audit['auxiliary_sensor_audit'])
    assert all(audit[k] is True for k in ('raw_sensor_reconstruction_pass','raw_model_command_replay_pass',
                                         'raw_command_audit_pass','model_state_unchanged'))
    errors = np.asarray(audit['observed_pose_xy_errors_m'],float)
    pose = dict(samples=len(errors),median_m=float(np.median(errors)),maximum_m=float(errors.max()))
    verify(sources); verify_artifacts(root,ids)
    write_json(OUTPUT,dict(status='COMPLETED_NO_RGB_JEPA_ADAPTER_NATIVE_EVIDENCE_AUTHENTICATED',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),case=CASE,worker_terminal_sha256=WORKER_SHA,
        source_sha256=sources,original_source_count=len(launch['source_sha256']),source_count=len(sources),
        artifact_sha256=ids,artifact_count=len(ids),original_startup_reconstructed=True,
        original_full_native_evaluation_reconstructed=True,original_readout_reconstructed=True,
        original_raw_audit_receipts_authenticated=True,original_raw_sensor_model_audit_reexecuted=False,
        training_input_admission_rerun=False,new_native_execution=False,new_model_inference=False,
        model_state_sha256=worker['model_state_sha256'],selected_actions=dict(actions),
        observations=count,first_terminal=terminal,readout=readout,observed_position_error=pose,
        verified_round_trip=worker['verified_round_trip'],goal_achieved=False))
    print(json.dumps(dict(output=str(OUTPUT),sha256=digest(OUTPUT),observations=count,
        first_terminal_frame=None if terminal is None else terminal['frame'],
        failure=None if terminal is None else terminal['failure'],readout=readout,position_error=pose)),flush=True)


if __name__ == '__main__': main()
