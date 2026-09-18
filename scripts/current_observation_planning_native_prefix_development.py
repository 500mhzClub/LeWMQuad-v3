"""Admit all eleven saved decisions and compare their new native physical prefix."""
from itertools import islice
import numpy as np
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.measured_floor_transport_prefix_development import normalize_labels
from lewm.current_observation_planning_prefix_development import compare_step
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

FRAMES = 11
CHANGED = 10


def admit_prefix(root, result):
    if (result['status'] != 'CURRENT_OBSERVATION_PLANNING_PREFIX_V1_COMPLETE'
            or result['model_loaded'] is not True or result['model_training'] is not False
            or result['native_execution'] is not False):
        raise ValueError('completed nonexecuting learned planning-memory prefix required')
    report = result['report']
    expected = dict(case='full_jepa_novel_maze_00',frames=FRAMES,first_requested_command_difference=CHANGED,
        first_terminal_policy_difference=None,final_requested_command=[.16,0.,.45],
        prior_requested_command=[0.,0.,.45],final_terminal=None,
        unchanged_observed_and_executed_residual_state_exact=True,
        raw_model_forecast_comparisons=8,all_compared_raw_model_forecasts_exact=True,
        original_actual_commands_before_intervention_exact=True,
        stopped_at_first_command_or_terminal_difference=True,following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True,model_state_sha256=MODEL_STATE,model_state_unchanged=True,
        accumulated_planning_cells_queried=False,persistent_contact_history_retained=True,
        selector_scan_state_retained=True,memoryless_controller=False,unexecuted_outcomes_inferred=False)
    for key,value in expected.items():
        if report[key] != value or type(report[key]) is not type(value):
            raise ValueError('exact completed planning-memory intervention required: '+key)
    rows = list(read_rows(root))
    if len(rows) != FRAMES: raise ValueError('all eleven and only eleven saved decisions required')
    for i,row in enumerate(rows):
        d = row['decision']; c = row['comparison']
        if (row['tick'] != i or d['tick'] != i or row['public_input_arrays_unchanged'] is not True
                or d['terminal'] is not None or d['failure'] is not None
                or c['unchanged_observed_and_executed_residual_state_exact'] is not True
                or c['requested_command_changed'] is not (i == CHANGED) or c['terminal_changed'] is not False
                or c['raw_model_forecasts_compared'] is not (i >= 3)
                or c['raw_model_forecasts_exact'] is not (i >= 3)
                or (i < CHANGED and d['requested_command'] != row['original_requested_command'])):
            raise ValueError('saved decision differs from the complete causal prefix')
    if (rows[-1]['decision']['new_selection'] != report['changed_selection']
            or rows[-1]['decision']['requested_command'] != report['final_requested_command']
            or rows[-1]['original_requested_command'] != report['prior_requested_command']):
        raise ValueError('complete final prospective decision and summary must agree')
    return report


def compare(prior, current, prefix_root, report):
    if report['frames'] != FRAMES or report['first_requested_command_difference'] != CHANGED:
        raise ValueError('fixed eleven-observation intervention required')
    count = 750+50*CHANGED; hashes = []
    for directory in (prior,current):
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
            if any(len(archive[k]) < count for k in archive.files): raise ValueError('complete physical prefix required')
            hashes.append(fingerprint({k:archive[k][:count] for k in archive.files}))
    if hashes[0] != hashes[1]: raise ValueError('physics differs before the new planning-map command')
    tapes = [read_json(p,'command_tape.json') for p in (prior,current)]
    if (any(len(t) < FRAMES for t in tapes)
            or any(not tapes[j][i]['completed'] for j in (0,1) for i in range(CHANGED))
            or any(tapes[0][i]['requested_command'] != tapes[1][i]['requested_command'] for i in range(CHANGED))
            or tapes[0][CHANGED]['requested_command'] != report['prior_requested_command']
            or tapes[1][CHANGED]['requested_command'] != report['final_requested_command']):
        raise ValueError('actual native commands must match the prospective planning-map intervention')
    readers = [IntentReturnRGBDReplay(p) for p in (prior,current)]
    acquisitions = [read_json(p,'auxiliary_camera_audit.json') for p in (prior,current)]
    checked = forecasts = 0
    for i,(old,new,bound) in enumerate(zip(islice(read_rows(prior),FRAMES),
            islice(read_rows(current),FRAMES),islice(read_rows(prefix_root),FRAMES),strict=True)):
        if not old['tick'] == new['tick'] == bound['tick'] == i: raise ValueError('ordered complete decisions required')
        for row in (old,new):
            if row['observation_index'] != i or row['pre_sample_index'] != 749+50*i:
                raise ValueError('actual observation endpoint required')
        public = []
        for directory,reader,rows in zip((prior,current),readers,acquisitions,strict=True):
            p,d,f,now = reader.packet(i)
            image,auxiliary = packet(directory,i,p,public_acquisition(rows[i]),now_ns=now)
            public.append(fingerprint((p,d,f,auxiliary,image,now)))
        if public[0] != public[1]: raise ValueError('paired native public prefix differs')
        if new['decision'] != bound['decision']:
            raise ValueError('complete new native decision differs from prospective replay')
        check = compare_step(normalize_labels(old['decision']),new['decision'],tapes[0][i]['requested_command'],frame=i)
        if check != bound['comparison']: raise ValueError('native shared-state/model comparison differs from prospective replay')
        forecasts += int(check['raw_model_forecasts_compared'])
        for row,tape in zip((old,new),tapes,strict=True):
            if row['decision']['requested_command'] != tape[i]['requested_command']:
                raise ValueError('native requested command differs from recorded decision')
        checked += 1
    if checked != FRAMES or forecasts != 8: raise ValueError('complete eleven-observation/eight-forecast prefix required')
    return dict(common_prefix_frames=FRAMES,first_intervention_frame=CHANGED,
        physical_prefix_samples=count,raw_physics_prefix_sha256=hashes[0],
        physical_and_public_prefix_exact=True,shared_observed_and_executed_residual_state_exact=True,
        all_preintervention_requested_commands_exact=True,complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=forecasts,all_compared_raw_model_forecasts_exact=True,
        original_intervention_command=tapes[0][CHANGED]['requested_command'],
        candidate_intervention_command=tapes[1][CHANGED]['requested_command'],
        candidate_intervention_command_completed=tapes[1][CHANGED]['completed'],
        following_physical_outcomes_compared=False,unexecuted_outcomes_inferred=False)
