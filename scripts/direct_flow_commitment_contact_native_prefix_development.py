"""Authenticate controller recovery and its future exact physical intervention.

No launcher and no simulation authority: a future admitted native owner must
first wait for the existing native queue and admit all original model inputs.
"""
from contextlib import closing
from itertools import islice
import numpy as np
from scripts import replay_go2_contact_anchored_direct_flow_controller_prefix_v1 as replay
from scripts.navigation_artifact_root_development import artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.maze_decision_stream_development import read_rows

FRAMES = 562
INTERVENTION = 561
PHYSICS_SAMPLES = 750+50*INTERVENTION
LAUNCH_SHA = '6c577445f7bff2e58aad960c6d26908c683a20b0d7a4fbb2344967b5072da9ad'
OWNER = dict(pid=2822758, created=1789119437.63, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', replay.SOURCE])


def boundary(report):
    expected = dict(frames=FRAMES, exact_candidate_original_decisions=INTERVENTION,
        complete_original_controller_decisions_reconstructed=True, two_fresh_original_model_copies=True, original_forecasts_compared=558,
        boundary_terminal=None, boundary_failure=None, model_state_sha256=replay.MODEL_SHA,
        model_state_unchanged=True, following_recorded_observations_consumed=False,
        actual_original_commands_before_intervention_exact=True, new_command_executed=False,
        original_failed_outcome_preserved=True, native_execution=False, navigation_qualified=False, goal_achieved=False)
    if any(report.get(k) != v or type(report.get(k)) is not type(v) for k,v in expected.items()):
        raise ValueError('complete recovered fixed full-supervised contact-scoring controller prefix required')
    check = report['boundary_comparison']
    if check.get('stop') is not True or check.get('full_controller_recovered') is not True:
        raise ValueError('full controller recovery required before a fresh native test')
    command = report['boundary_requested_command']
    if (type(command) is not list or len(command) != 3
            or any(type(x) not in (int,float) or not np.isfinite(x) for x in command)):
        raise ValueError('finite explicit candidate intervention command required')
    action = report['boundary_selected_action']
    expected_command = [0.,0.,0.] if action is None else list(replay.candidate_commands(action)[0])
    if (command != expected_command or check.get('requested_command_changed') is not (command != [0.,0.,0.])
            or check.get('complete_original_decision_exact') is not False
            or check.get('original_forecast_compared') is not False):
        raise ValueError('exact selected action and honest boundary-command comparison required')
    # Recovery can select a measured hold. Do not falsely label it movement.
    return FRAMES, INTERVENTION


def command_row(row, frame):
    if (type(row.get('tick')) is not int or row['tick'] != frame or row.get('completed') is not True
            or row['pre_sample_index'] != 749+50*frame or row['post_sample_index'] != 799+50*frame):
        raise ValueError('complete actual command and exact physical endpoints required')


def reconstruct(report, originals, prospective, original_tape, observed):
    boundary(report)
    if len(original_tape) < FRAMES: raise ValueError('complete original command prefix required')
    count = forecasts = 0; last = None
    for frame,(old,saved,visual) in enumerate(zip(originals,prospective,observed,strict=True)):
        if frame >= FRAMES: raise ValueError('no following recorded observation is admitted')
        if (old['tick'] != frame or old['observation_index'] != frame or old['pre_sample_index'] != 749+50*frame
                or saved['tick'] != frame or visual['tick'] != frame):
            raise ValueError('complete ordered original and prospective observations required')
        command = original_tape[frame]; command_row(command,frame)
        if (saved['original_requested_command'] != command['requested_command']
                or saved['public_input_arrays_unchanged'] is not True
                or saved['complete_original_decision_reconstructed'] is not True
                or saved['public_input_sha256'] != visual['public_packet_sha256']):
            raise ValueError('exact public packet identity and original requested command required')
        check = replay.compare(old['decision'],saved['decision'],command['requested_command'],visual['candidate'],frame=frame)
        if check != saved['comparison'] or check['stop'] is not (frame==INTERVENTION):
            raise ValueError('complete saved controller comparisons must reconstruct exactly')
        count += 1; forecasts += int(check['original_forecast_compared']); last=saved
    if count != FRAMES or forecasts != report['original_forecasts_compared']:
        raise ValueError('complete original forecast and observation population required')
    if (last['comparison'] != report['boundary_comparison']
            or last['decision']['requested_command'] != report['boundary_requested_command']
            or last['decision']['selected_action'] != report['boundary_selected_action']
            or last['decision']['terminal'] != report['boundary_terminal']
            or last['decision']['failure'] != report['boundary_failure']):
        raise ValueError('boundary summary must agree with saved full-controller evidence')
    return dict(frames=count,first_intervention_frame=INTERVENTION,original_forecasts_compared=forecasts,
        complete_saved_comparisons_reconstructed=True)


def public_packets(directory):
    reader = replay.observer.IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory,'auxiliary_camera_audit.json')
    for frame in range(FRAMES):
        p,d,f,now = reader.packet(frame)
        image,aux = replay.observer.packet(directory,frame,p,replay.observer.public_acquisition(acquisitions[frame]),now_ns=now)
        yield replay.observer.fingerprint((p,d,f,image,aux))


def admit_prefix(result_sha, sources):
    replay.owners_ended()
    if replay.owner_live(OWNER): raise ValueError('original contact controller replay is still live')
    root = replay.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink(): raise ValueError('failed controller prefix cannot authorize a native test')
    verify_artifacts(root,{'result.json':result_sha,'launch.json':LAUNCH_SHA})
    result = read_json(root,'result.json'); launch = read_json(root,'launch.json')
    expected_artifacts = {'launch.json','context_decisions.jsonl.gz','report.json'}
    if (result['status'] != 'CONTACT_ANCHORED_DIRECT_FLOW_CONTROLLER_PREFIX_V1_COMPLETE'
            or result['native_execution'] is not False or set(result['artifact_sha256']) != expected_artifacts
            or result['artifact_sha256']['launch.json'] != LAUNCH_SHA
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or launch['case'] != list(replay.CASE) or launch['model_state_sha256'] != replay.MODEL_SHA
            or launch['boundary_frame'] != INTERVENTION
            or launch['implementation_class'] != 'DirectFlowCommitmentContactController'
            or launch['original_completed_worker_admitted'] is not True
            or launch['two_fresh_original_model_copies'] is not True
            or launch['actual_assigned_model_loader_required'] is not True
            or launch['full_training_ancestry_reexecuted'] is not False):
        raise ValueError('exact completed source-bound full-controller replay required')
    verify(result['source_sha256']); verify_artifacts(root,result['artifact_sha256'])
    if read_json(root,'report.json') != result['report']: raise ValueError('bound report must match terminal result')
    report = result['report']; boundary(report)
    if launch['observer_artifact_sha256'].get('result.json') != replay.OBSERVER_SHA:
        raise ValueError('exact completed observer predecessor required')
    verify_artifacts(replay.observer.OUTPUT,launch['observer_artifact_sha256'])
    expected_inputs = read_json(replay.observer.OUTPUT,'launch.json')['input_artifact_sha256']
    if launch['input_artifact_sha256'] != expected_inputs: raise ValueError('all original episode bindings required')
    if replay.verify_inputs(sources,launch['observer_artifact_sha256']) != expected_inputs:
        raise ValueError('original ended-worker admission must reconstruct')
    verify_artifacts(replay.native.OUTPUT,expected_inputs)
    prior = replay.native.OUTPUT/replay.CASE[0]
    with closing(read_rows(prior)) as old, closing(read_rows(root)) as saved, closing(read_rows(replay.observer.OUTPUT)) as visual:
        reconstruct(report,islice(old,FRAMES),saved,read_json(prior,'command_tape.json'),visual)
    with closing(read_rows(root)) as saved:
        for actual,row in zip(public_packets(prior),saved,strict=True):
            if actual != row['public_input_sha256']: raise ValueError('original raw packet differs from full replay')
    verify_artifacts(root,result['artifact_sha256'])
    if replay.owner_live(OWNER): raise ValueError('original replay unexpectedly live')
    return report


def executed_boundary(tapes,report):
    boundary(report)
    if len(tapes) != 2 or any(len(t) < FRAMES for t in tapes):
        raise ValueError('two complete actual command prefixes required')
    for tape in tapes:
        for frame in range(FRAMES): command_row(tape[frame],frame)
    if (any(tapes[0][i]['requested_command'] != tapes[1][i]['requested_command'] for i in range(INTERVENTION))
            or tapes[0][INTERVENTION]['requested_command'] != [0.,0.,0.]
            or tapes[1][INTERVENTION]['requested_command'] != report['boundary_requested_command']):
        raise ValueError('exact original commands and completed candidate intervention required')


def compare(prior,current,prefix_root,report):
    boundary(report); hashes=[]
    for directory in (prior,current):
        with np.load(artifact_path(directory.parent,directory.name+'/physics_trace.npz'),allow_pickle=False) as raw:
            if not raw.files or any(len(raw[k]) < PHYSICS_SAMPLES+50 for k in raw.files):
                raise ValueError('all shared preintervention physics and 50 completed boundary-command samples required')
            hashes.append(replay.observer.fingerprint({k:raw[k][:PHYSICS_SAMPLES] for k in raw.files}))
    if hashes[0] != hashes[1]: raise ValueError('physical trajectory differs before intervention')
    tapes=[read_json(p,'command_tape.json') for p in (prior,current)]; executed_boundary(tapes,report)
    with closing(read_rows(prior)) as old, closing(read_rows(prefix_root)) as saved, closing(read_rows(replay.observer.OUTPUT)) as visual:
        reconstructed=reconstruct(report,islice(old,FRAMES),saved,tapes[0],visual)
    count=0
    with closing(read_rows(current)) as actual, closing(read_rows(prefix_root)) as prospective:
        for frame,(new,saved,p,q) in enumerate(zip(islice(actual,FRAMES),prospective,
                public_packets(prior),public_packets(current),strict=True)):
            if (frame >= FRAMES or new['tick'] != frame or new['observation_index'] != frame
                    or new['pre_sample_index'] != 749+50*frame or p != q or p != saved['public_input_sha256']
                    or new['decision'] != saved['decision']
                    or new['decision']['requested_command'] != tapes[1][frame]['requested_command']):
                raise ValueError('complete physical/public/controller prefix must match prospective replay')
            count+=1
    if count != FRAMES: raise ValueError('complete actual intervention observation required')
    return dict(common_prefix_frames=count,first_intervention_frame=INTERVENTION,physical_prefix_samples=PHYSICS_SAMPLES,
        raw_physics_prefix_sha256=hashes[0],physical_and_public_prefix_exact=True,
        all_preintervention_requested_commands_exact=True,complete_candidate_decisions_match_prospective_prefix=True,
        original_forecasts_compared=reconstructed['original_forecasts_compared'],
        candidate_intervention_command=report['boundary_requested_command'],candidate_intervention_command_completed=True,
        intervention_command_changed=report['boundary_requested_command'] != [0.,0.,0.],
        recovered_observation_frame=INTERVENTION, boundary_command_samples_present=50,
        observer_and_full_controller_intervention=True,following_physical_outcomes_compared=False,
        navigation_verified=False,unexecuted_outcomes_inferred=False)
