"""Prospective saved-input comparison through the fixed first continuation command."""
from copy import deepcopy
from datetime import datetime, timezone
from itertools import islice
import hashlib
import json
import numpy as np
from lewm.hold_reorientation_development import HoldReorientation
from lewm.sustained_hold_reorientation_development import SustainedHoldReorientation
from scripts import diagnose_go2_completed_hold_reorientation_maze02_v1 as prior
from scripts.maze_decision_stream_development import read_rows, NAME
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify, digest, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE = 'scripts/check_go2_sustained_hold_reorientation_saved_prefix_v1.py'
PROTOCOL = 'docs/go2_sustained_hold_reorientation_v1_2026-09-11.md'
TEST = 'lewm/tests/test_sustained_hold_reorientation_development.py'
CONTROLLER = 'lewm/sustained_hold_reorientation_controller_development.py'
PRIOR_SHA = '8b85fcb4a5c58706ad84604ebcfd5885d94b831ef9110e528d8a1a0991c02cf2'
OUTPUT = ROOT/'docs/go2_sustained_hold_reorientation_saved_prefix_2026-09-11.json'
FRAMES = 407


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def main():
    if digest(prior.OUTPUT) != PRIOR_SHA: raise ValueError('exact completed hold diagnosis required')
    previous = json.loads(prior.OUTPUT.read_text())
    sources = discover_sources((SOURCE,PROTOCOL,TEST,CONTROLLER), previous['source_sha256'] |
        {str(prior.OUTPUT.relative_to(ROOT)):PRIOR_SHA})
    verify(sources); root=prior.wait.native.OUTPUT
    verify_artifacts(root,{'result.json':prior.NATIVE_SHA,'launch.json':prior.NATIVE_LAUNCH})
    result=read_json(root,'result.json');record=result['conditions'][0];name=record['case'];directory=root/name
    ids={n:result['artifact_sha256'][n] for n in (name+'/'+NAME,name+'/command_tape.json')}
    verify_artifacts(root,ids);tape=read_json(directory,'command_tape.json')
    old=HoldReorientation();new=SustainedHoldReorientation();comparisons=[];first=None;boundary=None
    for row in islice(read_rows(directory),FRAMES):
        frame=row['tick'];d=row['decision'];saved=d['new_selection']
        if frame != len(comparisons) or d['terminal'] is not None or d['tick'] != frame:
            raise ValueError('uninterrupted admitted original prefix required')
        candidate=saved
        if saved is not None:
            inherited=deepcopy(saved);event=inherited.pop('hold_reorientation',None)
            if event:
                inherited.update(action=event['original_action'],action_index=0,requested_command=[0.,0.,0.])
                if event['original_action']!='hold':raise ValueError('exact original hold intervention required')
            now=1_500_000_000+frame*100_000_000
            reconstructed=old.reconsider(inherited,frame=frame,now_ns=now)
            if reconstructed != saved:raise ValueError('entire original helper selection must reconstruct: '+str(frame))
            pose=d['evidence']['current_pose'];B=np.asarray(d['memory_receipt']['map_from_initial'])
            R=np.asarray(pose['rotation_initial_body_from_current_body'])
            if pose['frame']!=frame:raise ValueError('current admitted observed heading required')
            candidate=new.reconsider(inherited,frame=frame,now_ns=now,observed_heading_map=(B@R)[:2,0])
        old_command=d['requested_command'];new_command=old_command if candidate is None else candidate['requested_command']
        command=tape[frame]
        if (not command['completed'] or command['tick']!=frame or command['requested_command']!=old_command
                or command['pre_sample_index']!=749+50*frame or command['post_sample_index']!=799+50*frame):
            raise ValueError('original completed command prefix required')
        changed=new_command!=old_command
        if changed and first is None:first=frame
        comparisons.append(dict(frame=frame,original_row_sha256=identity(row),
            original_selection_sha256=identity(saved),candidate_selection_sha256=identity(candidate),
            original_requested_command=old_command,candidate_requested_command=new_command,changed=changed))
        if frame==FRAMES-1:
            boundary=dict(frame=frame,original_selection=saved,candidate_selection=candidate,
                original_requested_command=old_command,candidate_requested_command=new_command)
    if (len(comparisons)!=FRAMES or first!=406 or sum(r['changed'] for r in comparisons)!=1
            or boundary['original_requested_command']!=[0.,0.,0.]
            or boundary['candidate_requested_command']!=[0.,0.,.45]):
        raise ValueError('fixed first sustained left-turn request must be observation 406')
    verify(sources);verify_artifacts(root,ids|{'result.json':prior.NATIVE_SHA,'launch.json':prior.NATIVE_LAUNCH})
    write_json(OUTPUT,dict(status='SUSTAINED_HOLD_REORIENTATION_SAVED_PREFIX_COMPLETE',
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,source_count=len(sources),
        prior_diagnosis_sha256=PRIOR_SHA,native_result_sha256=prior.NATIVE_SHA,
        original_input_sha256=ids,frames=FRAMES,first_changed_requested_command_frame=first,
        original_helper_selections_reconstructed=True,comparisons=comparisons,first_boundary=boundary,
        candidate_post_intervention_observations_consumed=False,
        candidate_changed_command_executed=False,full_controller_replayed=False,
        model_inference_reexecuted=False,full_native_artifact_rehash_reexecuted=False,
        native_execution=False,navigation_qualified=False,real_time_qualified=False,
        hardware_qualified=False,goal_achieved=False))
    print('SUSTAINED_SAVED_PREFIX_VERIFIED',digest(OUTPUT),len(sources),first,flush=True)


if __name__=='__main__':main()
