"""Require the actual settled-boundary pilot to reproduce its raw replay prefix."""
from itertools import islice
import numpy as np
from lewm.settled_boundary_prefix_comparison_development import compare_current
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def compare(prior, current, prefix_root, prefix_report):
    frames = prefix_report['frames']; changed = prefix_report['first_mission_behavior_difference']
    if type(frames) is not int or type(changed) is not int or changed < 0 or frames != changed+1:
        raise ValueError('bound first mission-state intervention prefix required')
    count = 750+50*changed; hashes = []
    for directory in (prior, current):
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as z:
            if any(len(z[k]) < count for k in z.files):
                raise ValueError('complete physical prefix required')
            hashes.append(fingerprint({k:z[k][:count] for k in z.files}))
    if hashes[0] != hashes[1]:
        raise ValueError('physics differs before the observed mission intervention')
    tapes = [read_json(p,'command_tape.json') for p in (prior,current)]
    if (any(len(t)<frames for t in tapes)
            or any(tapes[0][i]['requested_command'] != tapes[1][i]['requested_command'] for i in range(frames))
            or tapes[1][changed]['requested_command'] != prefix_report['final_requested_command']):
        raise ValueError('all prefix commands including transition hold must remain exact')
    readers=[IntentReturnRGBDReplay(p) for p in (prior,current)]
    acquisitions=[read_json(p,'auxiliary_camera_audit.json') for p in (prior,current)]
    checked=0
    for i,(old,new,bound) in enumerate(zip(islice(read_rows(prior),frames),
            islice(read_rows(current),frames),islice(read_rows(prefix_root),frames),strict=True)):
        if not old['tick']==new['tick']==bound['tick']==i:
            raise ValueError('ordered complete prefix identities required')
        public=[]
        for directory,reader,rows in zip((prior,current),readers,acquisitions,strict=True):
            p,d,f,now=reader.packet(i)
            auxiliary=packet(directory,i,p,public_acquisition(rows[i]),now_ns=now)
            public.append(fingerprint((p,d,f,auxiliary,now)))
        if public[0]!=public[1]:raise ValueError('public input differs before mission intervention')
        if new['decision']!=bound['decision']:
            raise ValueError('complete native decision differs from bound controller replay')
        comparison=compare_current(old['decision'],new['decision'])
        if bool(comparison['mission_behavior_differences'])!=(i==changed) or comparison['requested_command_changed']:
            raise ValueError('exact first mission-state change with unchanged hold required')
        if any(row['decision']['requested_command']!=tape[i]['requested_command']
                for row,tape in zip((old,new),tapes,strict=True)):
            raise ValueError('saved decisions differ from actual command tape')
        checked+=1
    if checked!=frames:raise ValueError('complete prospective prefix required')
    return dict(common_prefix_frames=frames,first_changed_mission_behavior=changed,
        first_changed_command_in_compared_prefix=None,raw_physics_prefix_sha256=hashes[0],
        physical_and_public_prefix_exact=True,all_prefix_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,
        only_declared_settling_mission_state_differs=True,
        observation_before_intervention_included=True,unexecuted_outcomes_inferred=False)
