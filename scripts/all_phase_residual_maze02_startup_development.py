"""Pair only the pre-command physics and public startup, not model forecasts."""
from itertools import islice
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import artifact_path
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def admit_startup(physics, tapes, decisions, public):
    if not all(len(items)==2 for items in (physics, tapes, decisions, public)):
        raise ValueError('one original and one candidate startup required')
    hashes=[]
    for trace, tape, rows, packets in zip(physics,tapes,decisions,public,strict=True):
        if not trace or any(len(v)<900 for v in trace.values()):
            raise ValueError('all900 original pre-command physics samples required')
        if len(tape)<3 or len(rows)!=4 or len(packets)!=4:
            raise ValueError('three complete warmup commands and four public observations required')
        for i,row in enumerate(rows):
            if row['tick']!=i or row['observation_index']!=i or row['pre_sample_index']!=749+50*i:
                raise ValueError('ordered exact public observation endpoints required')
            if i<3:
                command=tape[i]
                if (command['tick']!=i or command['completed'] is not True
                        or command['requested_command']!=[0.,0.,0.]
                        or row['decision']['requested_command']!=[0.,0.,0.]
                        or command['pre_sample_index']!=749+50*i
                        or command['post_sample_index']!=799+50*i):
                    raise ValueError('three actually completed zero warmup commands required')
        hashes.append(fingerprint({k:v[:900] for k,v in trace.items()}))
    if hashes[0]!=hashes[1]: raise ValueError('native physics differs before any learned command')
    if public[0]!=public[1]: raise ValueError('public startup differs before any learned command')
    return dict(common_prefix_frames=4, physical_prefix_samples=900,
        raw_physics_prefix_sha256=hashes[0], public_startup_sha256=fingerprint(public[0]),
        physical_and_public_startup_exact=True, completed_zero_warmup_commands=3,
        original_first_model_command=decisions[0][3]['decision']['requested_command'],
        candidate_first_model_command=decisions[1][3]['decision']['requested_command'],
        original_first_model_terminal=decisions[0][3]['decision']['terminal'],
        candidate_first_model_terminal=decisions[1][3]['decision']['terminal'],
        model_forecasts_required_equal=False, post_warmup_controller_state_required_equal=False,
        later_physical_outcomes_compared=False, unexecuted_outcomes_inferred=False)


def compare_startup(prior, current):
    physics=[]; tapes=[]; decisions=[]; public=[]
    for directory in (prior,current):
        path=artifact_path(directory.parent,directory.name+'/physics_trace.npz')
        with np.load(path,allow_pickle=False) as saved:
            physics.append({k:saved[k][:900].copy() for k in saved.files})
        tapes.append(read_json(directory,'command_tape.json'))
        decisions.append(list(islice(read_rows(directory),4)))
        reader=IntentReturnRGBDReplay(directory)
        acquisitions=read_json(directory,'auxiliary_camera_audit.json'); packets=[]
        for i in range(4):
            p,d,f,now=reader.packet(i)
            image,auxiliary=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
            packets.append(fingerprint((p,d,f,auxiliary,image,now)))
        public.append(packets)
    return admit_startup(physics,tapes,decisions,public)
