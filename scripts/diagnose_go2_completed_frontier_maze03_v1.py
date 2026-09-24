"""Authenticate the completed frontier diagnostic and inspect its full decisions."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts import await_go2_reached_frontier_maze03_native_v1 as wait
from scripts.maze_decision_stream_development import read_rows
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,verify,digest,write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE='scripts/diagnose_go2_completed_frontier_maze03_v1.py'
WAIT_SHA='5c6ed7f4944b375e4eb7db461ed7f5f5a05015942645d9e618e895fff4a5680b'
WAIT_LAUNCH='68c10ea5a869d6236975372a525dc4586ba7ba16cbeefb17ff0fbb2b57c07a74'
NATIVE_SHA='262fbd020dd2407e26de2a7801b53462655bf47afb9075bd9df7dd7af52352db'
NATIVE_LAUNCH='cb44a0bc2ca68663ebc983e9d5fdbbe6b8c213b3e480240b93082c26695d371b'
OWNER=dict(pid=2663938,created=1789032169.81,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python',wait.SOURCE])
OUTPUT=ROOT/'docs/go2_completed_frontier_maze03_diagnosis_2026-09-11.json'


def main():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=wait.BOOT or wait.owner_live(OWNER):
        raise ValueError('original frontier waiter must have ended on the recorded boot')
    for root in (wait.OUTPUT,wait.native.OUTPUT):
        if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
            raise ValueError('original failure must not be replaced by a completed readout')
    verify_artifacts(wait.OUTPUT,{'result.json':WAIT_SHA,'launch.json':WAIT_LAUNCH})
    saved=read_json(wait.OUTPUT,'result.json');registration=read_json(wait.OUTPUT,'launch.json')
    sources=discover_sources((SOURCE,),registration['source_sha256']);verify(sources)
    verify_artifacts(wait.OUTPUT,saved['artifact_sha256'])
    if (saved['status']!='REACHED_FRONTIER_MAZE03_NATIVE_WAIT_COMPLETE'
            or saved['source_sha256']!=registration['source_sha256']
            or registration['boot_id']!=wait.BOOT or registration['waiter_pid']!=OWNER['pid']):
        raise ValueError('exact completed original waiter and source identities required')
    inputs=read_json(wait.OUTPUT,'input_completion.json')
    reconstructed=wait.authenticate_completed(sources,inputs)
    if reconstructed!=saved['report'] or reconstructed!=read_json(wait.OUTPUT,'native_completion.json'):
        raise ValueError('complete original waiter/native completion must reconstruct')
    root=wait.native.OUTPUT
    verify_artifacts(root,{'result.json':NATIVE_SHA,'launch.json':NATIVE_LAUNCH})
    result=read_json(root,'result.json');launch=read_json(root,'launch.json')
    record=result['conditions'][0];name=record['case'];directory=root/name
    audit=read_json(root,name+'_audit.json');collection=read_json(directory,'result.json')
    with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
        contact=archive['physics_contact']
        if len(contact)!=collection['physics_samples']:raise ValueError('complete physics contact population required')
        readout=wait.native.case_readout(audit,collection,contact)
    if readout!=record['readout'] or collection!=record['collection']:
        raise ValueError('actual original collection and native readout must reconstruct')
    terminal=collection['mission_receipt']['frame']
    compact=[];events=[];modes=Counter();actions=Counter();infeasible=[];start_conflicts=[]
    stream_hash=hashlib.sha256();count=0
    for row in read_rows(directory):
        frame=row['tick'];decision=row['decision']
        stream_hash.update(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n')
        if frame!=count:raise ValueError('complete ordered decision history required')
        count+=1
        if frame>terminal:
            if decision['tick']!=terminal or decision['requested_command']!=[0.,0.,0.]:
                raise ValueError('original frozen terminal decision and zero-command drain required')
            continue
        if decision['tick']!=frame:raise ValueError('original live decision frame must match observation')
        selection=decision['new_selection'];transition=decision['last_frontier_transition_receipt']
        modes[decision['planner_mode']]+=1;actions[str(decision['selected_action'])]+=1
        proposal={} if selection is None else selection.get('proposal',{})
        checks=[] if selection is None else selection.get('nominal_action_checks',[])
        start=proposal.get('start_clearance')
        row_summary=dict(frame=frame,mode=decision['planner_mode'],action=decision['selected_action'],
            requested_command=decision['requested_command'],terminal=decision['terminal'],
            consecutive_infeasible_observations=decision['consecutive_infeasible_observations'],
            frontier_transition=transition,proposal_status=proposal.get('status'),start_clearance=start,
            route_cell_count=len(proposal.get('route_cells',[])),
            no_surface_conflict_candidates=None if selection is None else selection.get('no_surface_conflict_candidates'),
            phase_admissible_candidates=None if selection is None else selection.get('phase_admissible_candidates'),
            phase_allowed_actions=None if selection is None else selection.get('phase_allowed_actions'),
            nominal_action_checks=checks)
        compact.append(row_summary)
        if transition and transition['frame']==frame and (transition['reached_frontier_cell'] is not None
                or transition['released_frontier_cells'] or transition['mission_goal_changed']):events.append(transition)
        if checks and all(c['nominal_disk_connector_clear'] is False for c in checks):infeasible.append(frame)
        if start and start['nominal_disk_connector_clear'] is False:start_conflicts.append(frame)
    if count!=collection['decisions'] or count!=collection['rgbd_frames'] or count-terminal-1!=10:
        raise ValueError('complete collected observations and ten terminal drain observations required')
    first=record['prefix_comparison']['first_intervention_frame']
    if not any(e['frame']==first and e['reached_frontier_cell'] is not None for e in events):
        raise ValueError('actual reached-frontier intervention must be in the completed stream')
    verify(sources);verify_artifacts(root,result['artifact_sha256']|{'result.json':NATIVE_SHA})
    verify_artifacts(wait.OUTPUT,saved['artifact_sha256']|{'result.json':WAIT_SHA})
    if wait.owner_live(OWNER):raise ValueError('original waiter unexpectedly live')
    write_json(OUTPUT,dict(status='COMPLETED_FRONTIER_NATIVE_DIAGNOSTIC_RECONSTRUCTED',
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,source_count=len(sources),
        native_source_count=len(result['source_sha256']),waiter_result_sha256=WAIT_SHA,
        native_result_sha256=NATIVE_SHA,native_launch_sha256=NATIVE_LAUNCH,
        complete_native_artifact_count=len(result['artifact_sha256']),original_owner_ended=True,
        original_completion_verifier_reexecuted=True,model_state_sha256=launch['model_state_sha256'],
        complete_decisions=count,live_decisions=terminal+1,terminal_drain_observations=10,
        canonical_decision_stream_sha256=stream_hash.hexdigest(),readout=readout,
        original_prefix_comparison=record['prefix_comparison'],frontier_events=events,
        modes=dict(modes),actions=dict(actions),all_nominal_actions_rejected_frames=infeasible,
        current_nominal_disk_conflict_frames=start_conflicts,decision_summary=compact,
        raw_sensor_model_audit_reexecuted=False,full_training_ancestry_reexecuted=False,
        original_physical_prefix_comparison_reexecuted=False,unexecuted_actions_evaluated=False,
        policy_selected=False,native_execution=False,new_sensor_data_collected=False,
        navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
    print('FRONTIER_DIAGNOSIS_VERIFIED',digest(OUTPUT),len(sources),count,flush=True)
    print('FRONTIER_EVENTS',[(e['frame'],e['reached_frontier_cell']) for e in events],flush=True)
    print('NOMINAL_REJECTIONS',infeasible,'START_CONFLICTS',start_conflicts,flush=True)


if __name__=='__main__':main()
