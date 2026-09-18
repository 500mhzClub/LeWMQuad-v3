"""Score saved forecasts against physics only for matching executed commands."""
import argparse
import json
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.terminal_translation_pulse_development import command_sequences, TRANSLATIONS
from lewm.physical_execution_development import rotation_xyzw
from scripts.navigation_artifact_root_development import BASE, validate_root
from scripts.evaluate_saved_motion_forecasts_development import metrics


def evaluate(root, *, start_frame=0):
    validate_root(root, must_exist=True)
    read = lambda name: json.loads((root/name).read_text())
    frames = {r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
    requests = {r['simulator_ns']:r['requested_command'] for r in read('requests.json')}
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as archive:
        poses = archive['base_pose_world'].copy()
    groups = {g:{h:{k:[] for k in ('raw','corrected')} for h in ('300','400','700','800','dispatch_to_700')}
        for g in ('all','pulse_translation','full_translation','turn','hold')}
    retained = []; inspected = 0
    for row in read('planning.json'):
        if row['frame'] < start_frame or 'motion_correction' not in row:
            continue
        inspected += 1
        frame = row['frame']; now = row['measured_ns']; i = ACTIONS.index(row['action'])
        pulse = bool(row['motion_correction'].get('terminal_translation_pulse', False))
        commands = command_sequences(row['committed_prefix'], pulse=pulse)[i]
        matches = [all(t in requests and np.allclose(requests[t],c,atol=1e-8,rtol=0)
            for t in range(now+j*100_000_000,now+(j+1)*100_000_000,20_000_000))
            for j,c in enumerate(commands)]
        valid = np.logical_and.accumulate(matches)
        group = ('pulse_translation' if pulse else 'full_translation') if row['action'] in TRANSLATIONS else 'hold' if row['action']=='hold' else 'turn'
        origin = poses[frames[frame]['physical_sample_index']]
        R = rotation_xyzw(origin[3:])
        predicted = {name:np.asarray(row['motion_correction'][key])[i]
            for name,key in (('raw','raw_forecast_xy_m'),('corrected','corrected_forecast_xy_m'))}
        horizons = []
        for h in (3,4,7,8):
            if not valid[h-1] or frame+h not in frames:
                continue
            actual = (R.T@(poses[frames[frame+h]['physical_sample_index'],:3]-origin[:3]))[:2]
            for name, forecast in predicted.items():
                error = forecast[h-1]-actual
                for label in ('all',group): groups[label][str(h*100)][name].append(error)
            horizons.append(h*100)
            if h==7:
                actual_start = (R.T@(poses[frames[frame+3]['physical_sample_index'],:3]-origin[:3]))[:2]
                for name, forecast in predicted.items():
                    error = (forecast[6]-forecast[2])-(actual-actual_start)
                    for label in ('all',group): groups[label]['dispatch_to_700'][name].append(error)
        if horizons: retained.append(dict(frame=frame,action=row['action'],group=group,matched_horizons_ms=horizons))
    return dict(start_frame=start_frame,inspected_selected_plans=inspected,
        actual_requested_command_prefix_required=True,actual_simulator_command_timestamps_used=True,
        short_translation_and_zero_tail_reconstructed=True,online_forecasts_reused_without_recomputation=True,
        native_state_evaluator_only=True,model_refit=False,counterfactual_outcomes_evaluated=False,
        results={g:{h:{name:metrics(errors) for name,errors in values.items()}
            for h,values in horizons.items()} for g,horizons in groups.items()},retained_windows=retained)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    parser.add_argument('--start-frame', type=int, default=0)
    args = parser.parse_args(); root = BASE/args.root_name
    report = evaluate(root, start_frame=args.start_frame)
    with (root/f'saved_pulse_forecast_evaluation_from{args.start_frame}_v1.json').open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report['results'],indent=2))
