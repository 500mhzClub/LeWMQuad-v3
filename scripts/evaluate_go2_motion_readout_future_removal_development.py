"""Remove visual change from frozen readouts on fixed training and maze panels."""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import evaluate_go2_all_motion_horizon_readout_development as panel

OUTPUT = panel.fit.OUTPUT / 'future_image_removal_v1'


@torch.inference_mode()
def main():
    base = panel.fit.OUTPUT
    fit_plan = json.loads((base/'plan.json').read_text())
    fit_result = json.loads((base/'result.json').read_text())
    assert fit_result['status'] == 'COMPLETE'
    sources = {'training':base/'training_translation_fit_diagnostic_v1/result.json',
               'maze':base/'maze00_evaluation/result.json'}
    paths = json.loads((base/'frame_paths.json').read_text())
    assert panel.fit.digest(base/'frame_paths.json') == fit_plan['input_sha256']['frame_paths.json']
    rows = []
    for population, source in sources.items():
        result = json.loads(source.read_text())
        assert result['status'] == 'COMPLETE'
        for row in result['rows']:
            if population == 'training':
                path = paths[row['pair'][0]]
                group = 'training_translation'
                reference = {k:row['predictions'][k] for k in ('starting_mixed',*panel.fit.ARMS)}
                identity = dict(sample_index=row['sample_index'],origin=row['origin'])
            else:
                path = str(panel.ROOT/'native'/f"rgb_{row['frame']:04d}.png")
                group = 'maze_'+row['group']
                reference = {k:row['predictions'][k+'_observed_future'] for k in ('starting_mixed',*panel.fit.ARMS)}
                identity = dict(frame=row['frame'])
            rows.append(dict(group=group,path=path,horizon_ms=row['horizon_ms'],actual=row['actual'],
                             observed_future_predictions=reference,**identity))
    needed = sorted({r['path'] for r in rows})
    receipts = {}
    for path in needed:
        resolved = Path(path).resolve()
        assert not any(p == 'sealed' or p.startswith('sealed_') for p in resolved.parts)
        assert resolved.is_file()
        marker = resolved.parent/'depth_retention.json'
        if str(marker) not in receipts:
            receipts[str(marker)] = json.loads(marker.read_text()) if marker.exists() else None
    assert panel.fit.digest(fit_plan['initial_checkpoint']) == fit_plan['initial_checkpoint_sha256']
    heads = {'starting_mixed':panel.fit.prior.previous.load('mixed_data')}
    for arm in panel.fit.ARMS:
        path = base/f'{arm}_final.pt'
        assert panel.fit.digest(path) == fit_result['checkpoint_sha256'][arm]
        state = torch.load(path,map_location='cpu',weights_only=False)
        head = panel.fit.prior.previous.load('mixed_data')
        head.load_state_dict(state['model_state_dict'])
        heads[arm] = head.eval().requires_grad_(False)
    OUTPUT.mkdir(exist_ok=False)
    panel.save(OUTPUT/'plan.json',dict(source_sha256=panel.fit.digest(__file__),
        source_results={k:dict(path=str(v),sha256=panel.fit.digest(v)) for k,v in sources.items()},
        checkpoint_sha256=fit_result['checkpoint_sha256'],starting_checkpoint_sha256=fit_plan['initial_checkpoint_sha256'],
        selection='unchanged completed 32-training-translation and 64-maze-window panels, both horizons',
        rows=len(rows),image_paths=needed,depth_retention_receipts=receipts,cpu_cores=[4,5,6,7],
        intervention='replace future features by current features; current input unchanged and feature difference exactly zero',
        no_training=True,no_predictor_inference=True,no_navigation=True,
        limitations=['Post-hoc information removal; identical-image pairs differ from moving training pairs.',
                     'Not a separately fitted current-only baseline or a causal proof of a learning shortcut.',
                     'Uses saved actual-future predictions; no identity subtraction or calibration is promoted.',
                     'Training examples are exposed; maze windows overlap on one exposed trajectory.']))
    started = time.monotonic()
    torch.set_num_threads(4)
    try:
        encoder = panel.VJepa21Arm()
        encoder.build(torch.device('cpu'),torch.float32)
        identity_predictions = {}
        for j,path in enumerate(needed):
            pixels = encoder.preprocess(path)[None]
            features = pool_tokens(F.layer_norm(encoder.tokens(pixels).float(),(1024,)))
            identity_predictions[path] = {name:head(features,features)[0].numpy() for name,head in heads.items()}
            if (j+1)%16 == 0 or j+1 == len(needed):
                print('FUTURE_REMOVAL_FEATURES',j+1,len(needed),flush=True)
        del encoder
        for row in rows:
            predictions = {'zero_motion':np.zeros(3)}
            for name in heads:
                predictions[name+'_current_only'] = identity_predictions[row['path']][name]
                predictions[name+'_observed_future'] = np.asarray(row['observed_future_predictions'][name])
            errors = {}
            for name,value in predictions.items():
                delta = value-np.asarray(row['actual'])
                delta[2] = np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                errors[name] = delta.tolist()
            row.update(predictions={k:v.tolist() for k,v in predictions.items()},errors=errors)
        summary = {g:{str(h):panel.metrics([r for r in rows if r['group']==g and r['horizon_ms']==h])
                      for h in (500,700)} for g in sorted({r['group'] for r in rows})}
        panel.save(OUTPUT/'result.json',dict(status='COMPLETE',rows=rows,by_group_horizon=summary,
            wall_s=time.monotonic()-started,plan_sha256=panel.fit.digest(OUTPUT/'plan.json'),
            automatically_promoted=False))
        print('FUTURE_REMOVAL_COMPLETE',json.dumps(summary),flush=True)
    except BaseException as error:
        panel.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
