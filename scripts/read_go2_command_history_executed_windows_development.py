"""Frozen command-history prediction on all completed RGB-pilot executions.

Public past-command histories and prospective plans are the only new predictor
inputs. Completed evaluator records supply truth after prediction. No refitting,
depth/RGB reads, native-artifact reads or alternative-policy outcome claims.
"""
import hashlib
import json
import time
import numpy as np
from scripts.fit_go2_local_motion_controls_development import BASE, OUTPUT as FIT, nominal
from scripts.read_go2_nominal_residual_executed_windows_development import metrics
from lewm.terminal_translation_pulse_development import command_sequences

OUTPUT = BASE/'go2_command_history_executed_windows_v1_attempt_001'


def command_history(arrays, frame, now):
    if frame < 3 or int(arrays['decision_ns'][frame]) != now:
        raise ValueError('four causal observations at the planning timestamp required')
    history = []
    for f in range(frame-3, frame+1):
        decision = arrays['decision_ns'][f]
        if decision != now+(f-frame)*100_000_000:
            raise ValueError('consecutive past observations required')
        measured = arrays['applied_command_measured_ns'][f]
        available = arrays['applied_command_available_ns'][f]
        if np.any(measured > decision) or np.any(available > decision):
            raise ValueError('future command history forbidden')
        age = np.where(measured >= 0, (decision-measured)/1e9, 1.5)[:, None]
        history.append(np.concatenate((arrays['applied_command_values'][f]/[.3, 1., .5],
            arrays['applied_command_valid'][f].astype(float), age), axis=1).ravel())
    result = np.concatenate(history)
    if result.shape != (420,) or not np.isfinite(result).all():
        raise ValueError('finite original four-frame command representation required')
    return result


def forecast(model, history, commands, horizon=6):
    base = nominal(commands)
    known = np.zeros((8, 3)); known[:horizon+1] = commands[:horizon+1]
    x = np.r_[known.ravel(), base[horizon], history]
    return base[horizon]+((x-model['mean'][horizon])/model['scale'][horizon]) @ model['coefficient'][horizon]+model['bias'][horizon]


def main():
    if OUTPUT.exists():
        raise ValueError('preserve executed command-history readout')
    identities = {}

    def read(path):
        if any(p == 'sealed' or p.startswith('sealed_') for p in path.resolve().parts):
            raise ValueError('protected input forbidden')
        data = path.read_bytes()
        identities[str(path.relative_to(BASE))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    fit = read(FIT/'result.json')
    if fit['status'] != 'COMPLETE':
        raise ValueError('completed frozen command-history fit required')
    if 'model_columns' in fit and fit['model_columns']['command_only'] != list(range(13, 460)):
        raise ValueError('original command-only feature order required')
    model_path = FIT/'command_only.npz'
    identities[str(model_path.relative_to(BASE))] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    with np.load(model_path, allow_pickle=False) as archive:
        model = {k: archive[k].copy() for k in ('mean', 'scale', 'coefficient', 'bias')}
    for key,shape in dict(mean=(8,447),scale=(8,447),coefficient=(8,447,3),bias=(8,3)).items():
        if model[key].shape!=shape or not np.isfinite(model[key]).all():
            raise ValueError('finite frozen command-only model required')
    assignments = read(BASE/'go2_neural_rgb_transfer_complete_comparison_v1_attempt_001/result.json')['rows']
    if len(assignments) != 36 or len({a['root_name'] for a in assignments}) != 36:
        raise ValueError('all fixed 36 assignments required')
    OUTPUT.mkdir()
    (OUTPUT/'launch.json').write_text(json.dumps(dict(roots=[a['root_name'] for a in assignments],
        model_sha256=identities[str(model_path.relative_to(BASE))], fit_changed=False,
        matched_horizon_ms=700, include_failed_assignment=True), indent=2)+'\n')
    started = time.monotonic(); rows = []
    try:
        for assignment in assignments:
            root = BASE/assignment['root_name']
            plans = {p['frame']:p for p in read(root/'planning.json') if 'selection' in p}
            archive_path = root/'native/policy_histories.npz'
            identities[str(archive_path.relative_to(BASE))] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            with np.load(archive_path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in ('decision_ns', 'applied_command_values',
                    'applied_command_valid', 'applied_command_measured_ns', 'applied_command_available_ns')}
            truth = read(root/'saved_executed_motion_forecast_evaluation_v1.json')
            yaw = read(root/'saved_neural_command_yaw_evaluation_v1.json')
            if any(d['matched_requested_sequence_through_ns'] != 700_000_000 for d in (truth, yaw)):
                raise ValueError('seven-tick actual execution matching required')
            yaw_rows = {r['frame']:r for r in yaw['rows']}
            for window in truth['rows']:
                frame = window['frame']; plan = plans[frame]; correction = plan['motion_correction']
                if window['action'] != plan['selection']['action']:
                    raise ValueError('selected/executed action mismatch')
                index = plan['selection']['action_index']
                commands = command_sequences(plan['committed_prefix'], pulse=correction['terminal_translation_pulse'])[index]
                history = command_history(arrays, frame, plan['measured_ns'])
                prediction = forecast(model, history, commands)
                base = nominal(commands)[6]
                learned_yaw = correction['upstream_prediction_for_yaw_ablation'][index][6]
                predictions = dict(command_history=prediction, command_integrated=base,
                    saved_corrected_neural=[*correction['learned_corrected_forecast_xy_m'][index][6],
                        float(np.arctan2(learned_yaw[2], learned_yaw[3]))],
                    saved_fitted_xy_command_yaw=[*correction['pose_command_forecast_xy_m'][index][6], base[2]])
                actual = window['actual_endpoint_xy_m']; angle = yaw_rows[frame]['actual_endpoint_yaw_rad']
                errors = {}
                for name, pred in predictions.items():
                    dyaw = pred[2]-angle
                    errors[name] = dict(xy_m=float(np.linalg.norm(np.asarray(pred[:2])-actual)),
                        yaw_rad=float(np.arctan2(np.sin(dyaw), np.cos(dyaw))))
                rows.append(dict(root_name=root.name, frame=frame, group=window['group'],
                    action=window['action'], prediction=prediction.tolist(), errors=errors))
            print('EXECUTED_COMMAND_READOUT', root.name, len(truth['rows']), flush=True)
        result = dict(status='COMPLETE', assignments=len(assignments), matched_windows=len(rows),
            pooled=metrics(rows),
            by_root={a['root_name']:metrics([r for r in rows if r['root_name']==a['root_name']]) for a in assignments},
            by_action_group={g:metrics([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})},
            input_sha256=identities, wall_s=time.monotonic()-started,
            scope=dict(independent_new_mazes=0, existing_development_mazes=2, overlapping_windows=True,
                refitted=False, matched_horizon_ms=700, native_artifacts_read=False, depth_read=False,
                truth_from_completed_evaluators_only=True, alternative_navigation_outcomes_established=False,
                yaw_truth='saved wrapped world-heading difference, as in prior executed-window readouts'))
        (OUTPUT/'rows.json').write_text(json.dumps(rows)+'\n')
        (OUTPUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('status','matched_windows','pooled','by_action_group','wall_s')}, indent=2))
    except Exception as error:
        (OUTPUT/'failure.json').write_text(json.dumps(dict(reason=repr(error)))+'\n')
        raise


if __name__ == '__main__':
    main()
