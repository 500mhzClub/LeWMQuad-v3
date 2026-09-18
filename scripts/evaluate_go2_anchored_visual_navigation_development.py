"""Frozen visual forecasts on all matched executed navigation windows."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lewm.rgb_body_tensor_interface_development import observation_tensors
from scripts import evaluate_go2_refitted_dynamics_navigation_forecasts_development as source
from scripts import train_go2_anchored_visual_dynamics_development as training

OUTPUT=training.OUTPUT/'navigation_visual_forecasts'
PLAN=Path('docs/go2_anchored_visual_navigation_plan_2026-09-17.json')
RESULT=Path('docs/go2_anchored_visual_navigation_result_2026-09-17.json')
ARMS=training.ARMS+('persistence','original_visual_jepa')


def prepare():
    assert not OUTPUT.exists()
    prior=json.loads(source.PLAN.read_text())
    assert sum(r['windows'] for r in prior['roots'])==2404
    training.probe.save(PLAN,dict(schema='anchored_visual_navigation.v1',roots=prior['roots'],arms=ARMS,
        primary='700ms frozen visual target MSE; all 2404 matched executed windows',
        horizons_ms=list(range(100,701,100)),all_four_runs_including_failures=True,
        source_sha256=source.digest(__file__),fit_sha256=source.digest(training.OUTPUT/'result.json'),
        no_fitting=True,targets_loaded_after_each_causal_forecast=True,
        common_frozen_target_space=True,new_navigation=False,overlapping_windows_not_independent=True,
        alternative_action_outcomes_not_measured=True))
    print('PREPARED 2404 common-space visual forecast windows',flush=True)


def metrics(rows):
    return dict(windows=len(rows),curves={arm:[dict(horizon_ms=100*(h+1),
        mse=float(np.mean([r['mse'][arm][h] for r in rows]))) for h in range(7)] for arm in ARMS} if rows else {})


@torch.inference_mode()
def run():
    plan=json.loads(PLAN.read_text())
    assert source.digest(__file__)==plan['source_sha256']
    assert source.digest(training.OUTPUT/'result.json')==plan['fit_sha256']
    torch.set_num_threads(1);cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    model=training.representation.load();predictors={a:training.load(a) for a in training.ARMS}
    OUTPUT.mkdir(exist_ok=False);started=time.monotonic();all_rows=[];summaries=[]
    for number,record in enumerate(plan['roots'],1):
        root=Path(record['root']);window_path=root/'saved_executed_motion_forecast_evaluation_v1.json'
        assert source.digest(window_path)==record['window_sha256']
        windows=json.loads(window_path.read_text())['rows']
        plans={p['frame']:p for p in json.loads((root/'planning.json').read_text()) if 'selection' in p}
        reader=source.NoisyPublicReplay(root/'native');packet=lru_cache(maxsize=16)(reader.policy_packet)
        rows=[]
        for ordinal,window in enumerate(windows,1):
            frame=window['frame'];p=plans[frame];c=p['motion_correction']
            history=source.causal_history_tensors([packet(i) for i in range(frame-3,frame+1)],p['measured_ns'])
            inputs=source.delayed_candidate_inputs(history,p['committed_prefix'],delay_ticks=3,commit_ticks=4)
            if c['terminal_translation_pulse']:
                inputs['known_action_blocks']=torch.as_tensor(source.command_sequences(p['committed_prefix'],pulse=True)[:,:,None],dtype=torch.float32)/torch.tensor([.3,1.,.5])
            action=source.ACTIONS.index(window['action'])
            # Score only the actually executed command plan. No future image
            # or sensor packet has been passed to either context encoder.
            actual_inputs={k:({n:v[action:action+1] for n,v in value.items()} if isinstance(value,dict)
                else value[action:action+1]) for k,value in inputs.items()}
            h=actual_inputs['observation_history']
            past=model.encoder({k:v.flatten(0,1) for k,v in h.items()}).reshape(1,4,32)
            anchor=model.target({'rgb':h['rgb'][:,-1]})
            args=(past,anchor,actual_inputs['known_action_blocks'],actual_inputs['known_action_valid'])
            predictions={a:predictor(*args)[0,:7].numpy().copy() for a,predictor in predictors.items()}
            predictions['persistence']=np.broadcast_to(anchor[0].numpy(),(7,32)).copy()
            predictions['original_visual_jepa']=model(**actual_inputs)['future_latents'][0,:7].numpy().copy()
            # Target-only side: future body/control are never supplied to the
            # visual target, and no future observation enters predictions.
            future=[]
            for offset in range(1,8):
                value=packet(frame+offset)
                assert value['image']['measured_ns']==p['measured_ns']+offset*100_000_000
                future.append(observation_tensors(value)['rgb'])
            target=model.target({'rgb':torch.stack(future)}).numpy()
            errors={a:np.square(v.astype(float)-target).mean(-1).tolist() for a,v in predictions.items()}
            rows.append(dict(run=number,frame=frame,action=window['action'],mse=errors))
            if ordinal%400==0:print('VISUAL_FORECASTS',number,ordinal,flush=True)
        summary=dict(run=number,root=str(root),total=metrics(rows),
                     by_action={a:metrics([r for r in rows if r['action']==a]) for a in source.ACTIONS})
        training.probe.save(OUTPUT/f'run_{number:02d}.json',dict(summary=summary,rows=rows))
        summaries.append(summary);all_rows.extend(rows)
        print('VISUAL_FORECAST_RUN_COMPLETE',number,len(rows),flush=True)
    assert len(all_rows)==2404
    result=dict(status='complete',plan_sha256=source.digest(PLAN),runs=summaries,total=metrics(all_rows),
        by_action={a:metrics([r for r in all_rows if r['action']==a]) for a in source.ACTIONS},
        common_frozen_visual_target=True,all_failed_returns_included=True,
        depth_and_physics_not_loaded=True,weights_unchanged=True,new_navigation=False,
        overlapping_windows_not_independent=True,wall_s=time.monotonic()-started)
    training.probe.save(OUTPUT/'result.json',result);training.probe.save(RESULT,result)
    print(json.dumps({a:c[-1] for a,c in result['total']['curves'].items()},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare() if args.prepare else run()
