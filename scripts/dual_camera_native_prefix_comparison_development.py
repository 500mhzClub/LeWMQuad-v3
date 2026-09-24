"""Require unchanged physical/public history and the exact prospective camera decision."""
from itertools import islice
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.dual_camera_json_prefix_comparison_development import compare_json_primary_decision
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def prefix_shape(report):
    frames=report['frames'];changed=report['first_auxiliary_intervention_frame']
    if (type(frames) is not int or type(changed) is not int or not 1<=changed<1883
            or frames!=changed+1 or report['exact_primary_decision_frames']!=changed):
        raise ValueError('complete first-camera-intervention prefix required')
    return frames,changed


def compare(prior, current, prefix_root, prefix_report):
    frames,changed=prefix_shape(prefix_report)
    count=750+50*changed;hashes=[]
    for directory in (prior,current):
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as z:
            if any(len(z[k])<count for k in z.files):raise ValueError('complete physical prefix required')
            hashes.append(fingerprint({k:z[k][:count] for k in z.files}))
    if hashes[0]!=hashes[1]:raise ValueError('physics differs before camera intervention')
    tapes=[read_json(p,'command_tape.json') for p in (prior,current)]
    if (any(len(t)<frames for t in tapes)
            or any(tapes[0][i]['requested_command']!=tapes[1][i]['requested_command'] for i in range(changed))
            or tapes[1][changed]['requested_command']!=prefix_report['final_requested_command']):
        raise ValueError('exact executed preintervention commands and bound intervention command required')
    readers=[IntentReturnRGBDReplay(p) for p in (prior,current)]
    acquisitions=[read_json(p,'auxiliary_camera_audit.json') for p in (prior,current)]
    checked=0
    for i,(old,new,bound) in enumerate(zip(islice(read_rows(prior),frames),
            islice(read_rows(current),frames),islice(read_rows(prefix_root),frames),strict=True)):
        if not old['tick']==new['tick']==bound['tick']==i:raise ValueError('ordered complete prefix required')
        public=[];current_packet=None
        for directory,reader,rows in zip((prior,current),readers,acquisitions,strict=True):
            p,d,f,now=reader.packet(i)
            image,auxiliary=packet(directory,i,p,public_acquisition(rows[i]),now_ns=now)
            public.append(fingerprint((p,d,f,auxiliary,image,now)))
            current_packet=(p,image,auxiliary,now)
        if public[0]!=public[1]:raise ValueError('paired public RGB/depth input differs before intervention')
        if new['decision']!=bound['decision']:raise ValueError('complete native decision differs from prospective replay')
        if i<changed:
            p,image,auxiliary,now=current_packet
            compare_json_primary_decision(old['decision'],new['decision'],p,image,auxiliary,now_ns=now)
        elif (not bound['first_auxiliary_intervention']
                or new['decision']['original_visual_evidence']['camera_selection']['auxiliary_attempted'] is not True):
            raise ValueError('actual first auxiliary intervention must match the bound decision')
        if any(row['decision']['requested_command']!=tape[i]['requested_command']
                for row,tape in zip((old,new),tapes,strict=True)):
            raise ValueError('saved decisions differ from actual tape')
        checked+=1
    if checked!=frames:raise ValueError('complete prospective prefix required')
    return dict(common_prefix_frames=frames,first_auxiliary_intervention_frame=changed,
        raw_physics_prefix_sha256=hashes[0],physical_and_public_prefix_exact=True,
        all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,
        observation_before_intervention_included=True,unexecuted_outcomes_inferred=False,
        original_intervention_command=tapes[0][changed]['requested_command'],
        candidate_intervention_command=tapes[1][changed]['requested_command'])
