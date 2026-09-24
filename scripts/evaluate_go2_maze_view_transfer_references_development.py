"""Frozen command references on the already fixed maze-view transfer windows."""
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from scripts.read_go2_command_history_executed_windows_development import command_history, forecast
from scripts.fit_go2_local_motion_controls_development import nominal

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
ROOT = BASE/'go2_maze_view_transfer_v1_attempt_001'
FIT = BASE/'go2_maze_view_readout_v1_attempt_003'
MODEL = BASE/'go2_short_pulse_command_control_v1_attempt_001'
OUTPUT = ROOT/'command_references'
KEYS = ('decision_ns', 'applied_command_values', 'applied_command_valid',
        'applied_command_measured_ns', 'applied_command_available_ns')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (OUTPUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def metrics(rows):
    result = {}
    for name in ('command_history', 'command_integrated', 'zero_motion'):
        error = np.asarray([r['errors'][name] for r in rows])
        result[name] = dict(xy_rmse_mm=float(np.sqrt(np.mean(np.sum(error[:, :2]**2, axis=1)))*1000),
            yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(error[:, 2]**2)))))
    return dict(windows=len(rows), metrics=result)


def main():
    # Add a separate reference result; never rewrite the frozen neural evaluation.
    assert not (FIT/'result.json').exists(), 'reference addition precedes fit outcome'
    plan = json.loads((ROOT/'plan.json').read_text())
    evaluation = json.loads((ROOT/'evaluation_plan.json').read_text())
    assert digest(ROOT/'transfer_targets.json') == evaluation['targets_sha256']
    assert json.loads((MODEL/'result.json').read_text())['status'] == 'COMPLETE'
    targets = json.loads((ROOT/'transfer_targets.json').read_text())
    with np.load(MODEL/'command_only.npz', allow_pickle=False) as archive:
        model = {k: archive[k].copy() for k in ('mean', 'scale', 'coefficient', 'bias')}
    for key, shape in dict(mean=(8,447), scale=(8,447), coefficient=(8,447,3), bias=(8,3)).items():
        assert model[key].shape == shape and np.isfinite(model[key]).all()
    OUTPUT.mkdir(exist_ok=False)
    save('plan.json', dict(source_sha256=digest(__file__), targets_sha256=evaluation['targets_sha256'],
        collection_plan_sha256=digest(ROOT/'plan.json'), model_sha256=digest(MODEL/'command_only.npz'),
        inputs='Four causal public applied-command histories and the previously fixed future requested-command tape.',
        targets='Unchanged full-body translation and relative-body yaw from the prepared neural evaluation.',
        future_measured_commands_used=False, refitted=False, fit_result_available=False,
        limitations=['Two independent mazes; overlapping windows.',
            'Actual-future image decoding and command forecasting receive different information.',
            'Descriptive component references, not navigation outcomes.']))
    try:
        arrays, identities, rows = {}, {}, []
        for case in sorted({r['case'] for r in targets}):
            path = ROOT/f'case_{case:02d}'/'policy_histories.npz'
            identities[str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as archive:
                arrays[case] = {k: archive[k].copy() for k in KEYS}
        for target in targets:
            frame, h = target['frame'], target['horizon_ms']//100-1
            data = arrays[target['case']]
            history = command_history(data, frame, int(data['decision_ns'][frame]))
            commands = np.asarray(plan['tape'][frame:frame+8], dtype=float)
            assert commands.shape == (8,3)
            predictions = dict(command_history=forecast(model, history, commands, h),
                command_integrated=nominal(commands)[h], zero_motion=np.zeros(3))
            errors = {}
            for name, prediction in predictions.items():
                error = prediction-np.asarray(target['actual'])
                error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
                errors[name] = error.tolist()
            rows.append(target | dict(predictions={k:v.tolist() for k,v in predictions.items()}, errors=errors))
        grouped = {str(m):{str(h):{g:metrics([r for r in rows if r['maze']==m
            and r['horizon_ms']==h and (g=='all' or r['group']==g)])
            for g in ('all','translation','turn','hold')} for h in (500,700)} for m in (0,1)}
        save('result.json', dict(status='COMPLETE', rows=rows, by_maze_horizon_group=grouped,
            command_history_input_sha256=identities, plan_sha256=digest(OUTPUT/'plan.json'),
            refitted=False, navigation_tested=False, targets_changed=False))
        print('MAZE_TRANSFER_REFERENCES_COMPLETE', json.dumps(grouped), flush=True)
    except BaseException as error:
        save('failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
