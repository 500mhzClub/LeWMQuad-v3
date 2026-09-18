"""Visual-target model on the same complete 2,404-window recorded population."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_refitted_dynamics_navigation_forecasts_development as base
from scripts import evaluate_go2_visual_target_jepa_development as study

OUTPUT = study.OUTPUT/'navigation_forecasts'
PLAN = Path('docs/go2_visual_target_navigation_forecast_plan_2026-09-17.json')
RESULT = Path('docs/go2_visual_target_navigation_forecast_result_2026-09-17.json')
VARIANTS = base.VARIANTS+('visual_target_jepa',)
metrics = bind(base.scoring.metrics,VARIANTS=VARIANTS)


def prepare():
    assert not OUTPUT.exists()
    previous = json.loads((base.OUTPUT/'result.json').read_text())
    assert previous['status']=='complete' and previous['total']['windows']==2404
    records = []
    for run in previous['runs']:
        path=base.OUTPUT/f"run_{run['run']:02d}.json"
        records.append(dict(run=run['run'],root=run['root'],previous=str(path),sha256=base.digest(path)))
    study.probe.save(PLAN,dict(schema='visual_target_navigation_forecast.v1',runs=records,
        source_sha256=base.digest(__file__),fit_sha256=base.digest(study.OUTPUT/'result.json'),
        variants=VARIANTS,windows=2404,retain_both_failed_returns=True,
        primary='700ms XY and yaw errors on all identical executed windows',
        previous_controls_reused_and_original_JEPA_recomputed=True,
        new_navigation=False,overlapping_windows_not_independent=True,no_fitting=True))
    print('PREPARED visual-target forecasts on all 2404 windows',flush=True)


@torch.inference_mode()
def run():
    plan=json.loads(PLAN.read_text())
    assert base.digest(__file__)==plan['source_sha256']
    assert base.digest(study.OUTPUT/'result.json')==plan['fit_sha256']
    torch.set_num_threads(1);cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    visual=study.load_readout('jepa');original=study.original.load_readout('jepa')
    OUTPUT.mkdir(exist_ok=False);started=time.monotonic();all_rows=[];summaries=[]
    for record in plan['runs']:
        assert base.digest(record['previous'])==record['sha256']
        old=json.loads(Path(record['previous']).read_text())['rows'];root=Path(record['root'])
        plans={p['frame']:p for p in json.loads((root/'planning.json').read_text()) if 'selection' in p}
        reader=base.NoisyPublicReplay(root/'native');packet=lru_cache(maxsize=8)(reader.policy_packet)
        with np.load(root/'native/physics_trace.npz',allow_pickle=False) as a:physics=a['base_pose_world']
        rows=[];new_rows=[]
        for ordinal,row in enumerate(old,1):
            frame=row['frame'];p=plans[frame];c=p['motion_correction']
            history=base.causal_history_tensors([packet(i) for i in range(frame-3,frame+1)],p['measured_ns'])
            inputs=base.delayed_candidate_inputs(history,p['committed_prefix'],delay_ticks=3,commit_ticks=4)
            if c['terminal_translation_pulse']:
                inputs['known_action_blocks']=torch.as_tensor(base.command_sequences(p['committed_prefix'],pulse=True)[:,:,None],dtype=torch.float32)/torch.tensor([.3,1.,.5])
            pred=visual(**inputs)['rollout_outcomes'].numpy()
            check=original(**inputs)['rollout_outcomes'].numpy()
            np.testing.assert_allclose(check[:,:,:4],np.asarray(c['upstream_prediction_for_yaw_ablation'])[:,:,:4],rtol=0,atol=2e-6)
            origin=physics[reader.noise_rows[frame]['physical_sample_index']]
            actual_poses=physics[[reader.noise_rows[frame+h]['physical_sample_index'] for h in range(1,8)]]
            actual_xy=((actual_poses[:,:3]-origin[:3])@base.rotation_xyzw(origin[3:]))[:,:2]
            actual=base.components(actual_xy);action=base.ACTIONS.index(row['action'])
            def errors(prediction):
                parts=base.components(prediction[action,:,:2])
                return {part:(1000*(np.asarray(parts[part])-actual[part])).tolist() for part in base.PARTS}
            check_errors=errors(check)
            for part in base.PARTS:
                np.testing.assert_allclose(check_errors[part],row['errors_mm']['original_jepa'][part],rtol=0,atol=1e-8)
            rotations=[base.rotation_xyzw(v[3:]) for v in (origin,actual_poses[-1])]
            actual_yaw=math.atan2(rotations[1][1,0],rotations[1][0,0])-math.atan2(rotations[0][1,0],rotations[0][0,0])
            delta=math.atan2(pred[action,6,2],pred[action,6,3])-actual_yaw
            new=dict(run=row['run'],frame=frame,action=row['action'],errors_mm=errors(pred),
                     yaw_error_rad=math.atan2(math.sin(delta),math.cos(delta)))
            new_rows.append(new)
            rows.append(row | dict(errors_mm=row['errors_mm'] | {'visual_target_jepa':new['errors_mm']},
                                   yaw_error_rad=row['yaw_error_rad'] | {'visual_target_jepa':new['yaw_error_rad']}))
            if ordinal%400==0:print('VISUAL_NAVIGATION_FORECAST',record['run'],ordinal,flush=True)
        summary=dict(run=record['run'],root=str(root),total=metrics(rows),
                     by_action={a:metrics([r for r in rows if r['action']==a]) for a in base.ACTIONS})
        study.probe.save(OUTPUT/f"run_{record['run']:02d}.json",dict(summary=summary,visual_rows=new_rows))
        summaries.append(summary);all_rows.extend(rows)
        print('VISUAL_NAVIGATION_FORECAST_RUN_COMPLETE',record['run'],len(rows),flush=True)
    assert len(all_rows)==2404
    result=dict(status='complete',plan_sha256=base.digest(PLAN),total=metrics(all_rows),runs=summaries,
        by_action={a:metrics([r for r in all_rows if r['action']==a]) for a in base.ACTIONS},
        original_JEPA_predictions_and_per_window_errors_reproduced=True,depth_not_loaded=True,
        failures_included=True,overlapping_windows_not_independent=True,new_navigation_executed=False,
        wall_s=time.monotonic()-started)
    study.probe.save(OUTPUT/'result.json',result);study.probe.save(RESULT,result)
    print(json.dumps(result['total']['models']['visual_target_jepa'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare() if args.prepare else run()
