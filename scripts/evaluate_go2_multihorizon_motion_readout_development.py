"""Fixed exposed-pilot transfer comparison for matched horizon training."""
import argparse
import json
import traceback

import numpy as np
import torch

from scripts import evaluate_go2_correlation_motion_readout_development as evaluation
from scripts import train_go2_multihorizon_motion_readout_development as training

PLAN = training.OUTPUT/'transfer_plan.json'
REFERENCE = training.OUTPUT/'command_history_reference.json'
OUTPUT = training.OUTPUT/'pilot_evaluation'
PREVIOUS = training.previous.OUTPUT/'pilot_evaluation/result.json'
digest = training.previous.original.digest


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def prepare():
    assert not PLAN.exists() and not REFERENCE.exists() and not OUTPUT.exists()
    old = json.loads(PREVIOUS.read_text())
    assert old['status'] == 'COMPLETE' and old['windows'] == 35
    plans = {r['frame']: r for r in json.loads((evaluation.ROOT/'planning.json').read_text())
        if 'selection' in r}
    rows = []
    for row in old['rows']:
        plan = plans[row['frame']]
        i = evaluation.ACTIONS.index(plan['action'])
        prediction = np.asarray(plan['motion_correction']['command_history_forecast_xy_yaw'])[i, row['horizon_ms']//100-1]
        delta = prediction-np.asarray(row['actual'])
        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        assert np.isfinite(delta).all()
        rows.append(dict(frame=row['frame'], horizon_ms=row['horizon_ms'], actual=row['actual'],
            predictions=dict(command_history=prediction.tolist()), errors=dict(command_history=delta.tolist())))
    assert len(rows) == 105 and len({r['frame'] for r in rows}) == 35
    by = {(r['frame'], r['horizon_ms']): r for r in rows}
    increments = []
    for frame in sorted({r['frame'] for r in rows}):
        delta = np.asarray(by[frame, 700]['errors']['command_history'])-by[frame, 300]['errors']['command_history']
        delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
        increments.append(dict(frame=frame, errors=dict(command_history=delta.tolist())))
    baseline = dict(status='COMPLETE', rows=rows, increments=increments,
        by_horizon={str(h): evaluation.metrics([r for r in rows if r['horizon_ms'] == h]) for h in (300, 500, 700)},
        commit_interval_300_to_700ms=evaluation.metrics(increments),
        no_new_inference=True, no_navigation=True, scope='same existing 35 matched executed pilot windows')
    save(REFERENCE, baseline)
    save(PLAN, dict(previous_result=str(PREVIOUS), previous_result_sha256=digest(PREVIOUS),
        command_history_reference_sha256=digest(REFERENCE),
        evaluator_sha256=digest(__file__), shared_evaluator_sha256=digest(evaluation.__file__),
        pilot=str(evaluation.ROOT), windows=sorted({r['frame'] for r in rows}),
        horizons_ms=[300, 500, 700], primary_horizon_ms=700,
        secondary='500-ms supervised endpoint, 300-ms committed prefix, and 300--700-ms interval',
        metrics=['XY RMSE in millimetres', 'yaw RMSE in degrees'],
        fixed_heads=['original', 'starting_mixed', *training.ARMS],
        future_inputs=['observed_future', 'action', 'no_future_action'],
        controls=['zero_motion', 'command_history'],
        checkpoint_selection='fixed final training step for both new heads',
        no_new_navigation=True, automatic_promotion=False,
        limitations=['previously exposed turn/hold pilot; 35 overlapping windows',
            'not independent navigation or translation-generalisation evidence',
            'post hoc experiment choice, population fixed before new training results']))
    print('MULTIHORIZON_TRANSFER_PREPARED', json.dumps(baseline['by_horizon']), flush=True)


def main():
    plan = json.loads(PLAN.read_text())
    assert digest(__file__) == plan['evaluator_sha256']
    assert digest(evaluation.__file__) == plan['shared_evaluator_sha256']
    assert digest(PREVIOUS) == plan['previous_result_sha256']
    assert digest(REFERENCE) == plan['command_history_reference_sha256']
    result = json.loads((training.OUTPUT/'result.json').read_text())
    assert result['status'] == 'COMPLETE' and result['steps'] == training.STEPS
    heads = dict(original=training.previous.original.load(), starting_mixed=training.previous.load('mixed_data'))
    metadata = dict(original=digest(training.previous.original.OUTPUT/'readout.pt'),
        starting_mixed=digest(training.previous.OUTPUT/'mixed_data_final.pt'))
    for arm in training.ARMS:
        path = training.OUTPUT/f'{arm}_final.pt'
        assert digest(path) == result['checkpoint_sha256'][arm]
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['updates'] == training.STEPS
        assert state['plan_sha256'] == digest(training.OUTPUT/'plan.json')
        head = training.previous.load('mixed_data')
        head.load_state_dict(state['model_state_dict'], strict=True)
        heads[arm] = head.eval().requires_grad_(False)
        metadata[arm] = digest(path)
    # Historical depth is not read; consult any replay-retirement markers first.
    retention = {}
    for name in ('depth_retention.json', 'native/depth_retention.json'):
        path = evaluation.ROOT/name
        if path.exists():
            retention[name] = json.loads(path.read_text())
    evaluation.main(heads=heads, output=OUTPUT, head_metadata=metadata)
    try:
        evaluated = json.loads((OUTPUT/'result.json').read_text())
        old = {(r['frame'], r['horizon_ms']): r for r in json.loads(PREVIOUS.read_text())['rows']}
        assert len(evaluated['rows']) == len(old) == 105
        for row in evaluated['rows']:
            reference = old[row['frame'], row['horizon_ms']]
            for arm in plan['future_inputs']:
                np.testing.assert_allclose(row['predictions']['starting_mixed_'+arm],
                    reference['predictions']['mixed_data_'+arm], rtol=0, atol=2e-5)
        baseline = json.loads(REFERENCE.read_text())
        save(OUTPUT/'comparison.json', dict(status='COMPLETE', transfer_plan_sha256=digest(PLAN),
            starting_mixed_predictions_reproduced_atol=2e-5, depth_retention_receipts=retention,
            by_horizon={h: evaluated['by_horizon'][h]|baseline['by_horizon'][h] for h in ('300', '500', '700')},
            commit_interval_300_to_700ms=evaluated['commit_interval_300_to_700ms']|baseline['commit_interval_300_to_700ms'],
            navigation_tested=False, independent_generalisation_tested=False, automatically_promoted=False))
        print('MULTIHORIZON_TRANSFER_COMPLETE', str(OUTPUT/'comparison.json'), flush=True)
    except BaseException as error:
        save(OUTPUT/'comparison_failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        main()
