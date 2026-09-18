"""Same-target predictions for three data successors and their predecessors."""
from collections import defaultdict
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.pulse_timed_dataset_development import stack_samples
from scripts.pre_switch_training_data_development import BASE, ROOTS, PacketReader, inputs
from scripts.observation_horizon_snapshot_development import load_snapshot
from scripts.train_go2_pre_switch_successor_development import OUTPUT as FITS, CONDITIONS
from scripts.prepare_go2_pre_switch_transfer_development import OUTPUT as TARGETS

OLD = BASE/'go2_all_phase_matched_fits_v1_attempt_001'
OUTPUT = BASE/'go2_pre_switch_transfer_comparison_v1_attempt_001'
PREDICTION_INTERPRETATION = 'absolute neural heads without external fitted corrections'


def read(path):
    return json.loads(path.read_text())


def load(directory, record):
    return load_snapshot(directory, record['filename'], sha256=record['sha256'],
        expected_binding=record['binding'], expected_config=record['configuration'])


def comparison_roster():
    roster = []
    for condition in CONDITIONS:
        roster.append(('augmented_'+condition,FITS,read(FITS/(condition+'_fit.json'))))
        result = read(OLD/'result.json')
        jobs = [j for w in result['records'] for j in w['jobs'] if j['name']==f'seed_2026091001_full_{condition}']
        if len(jobs)!=1: raise ValueError('one fixed predecessor')
        name = next(n for n in jobs[0]['artifact_sha256'] if n.endswith('_fit.json'))
        roster.append(('original_'+condition,OLD,read(OLD/name)['snapshot']))
    return roster


def summarize(prediction, rows, selected):
    report = {}
    for h in (3, 7, 8):
        ids = [i for i in selected if rows[i]['targets'][h-1]['motion_valid']]
        if not ids:
            report[str(h*100)] = dict(windows=0); continue
        truth = np.asarray([rows[i]['targets'][h-1]['motion'] for i in ids])
        p = prediction[ids, h-1]
        xy = np.linalg.norm(p[:, :2]-truth[:, :2], axis=1)
        yaw = np.arctan2(np.sin(p[:, 2]-truth[:, 2]), np.cos(p[:, 2]-truth[:, 2]))
        report[str(h*100)] = dict(windows=len(ids), xy_rmse_mm=1000*float(np.sqrt(np.mean(xy**2))),
            xy_p90_mm=1000*float(np.percentile(xy, 90)),
            yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(yaw**2)))))
    return report


@torch.inference_mode()
def main():
    if OUTPUT.exists():
        raise ValueError('preserve comparison')
    terminal = read(FITS/'result.json')
    if terminal['status'] != 'COMPLETE' or terminal['total_updates'] != 3600:
        raise ValueError('all final fixed-budget checkpoints required')
    torch.set_num_threads(1); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    torch.use_deterministic_algorithms(True)
    began = time.monotonic()
    rows = [r for r in read(TARGETS/'windows.json') if r['available']]
    # Original switch rows keep their causal clock/index fields in the receipt;
    # family and recovered rows additionally expose them at the top level.
    rows = [r | {k:r['observation_horizon_receipt'][v] for k,v in (
        ('decision_ns','departure_ns'), ('history_observation_indices','history_observation_indices'))}
        for r in rows]
    cache = {}; groups = defaultdict(list); policy_hashes = {}
    for i,r in enumerate(rows): groups[r['source'],r['trial']].append(i)
    for (source,trial), ids in sorted(groups.items()):
        directory = ROOTS[source]/trial
        frames = sorted({f for i in ids for f in rows[i]['history_observation_indices']})
        for name in ['policy_observations.json','policy_histories.npz']+[f'rgb_{f:04d}.png' for f in frames]:
            p = directory/name; policy_hashes[str(p.relative_to(BASE))] = hashlib.sha256(p.read_bytes()).hexdigest()
        packets = {f:load_route_observation(directory,f) for f in frames}
        for i in ids:
            reader = PacketReader(packets, rows[i]['history_observation_indices'])
            cache[i] = inputs(rows[i], reader)
            if reader.requests != rows[i]['history_observation_indices']:
                raise ValueError('inference reads only the four causal packets')
    selections = dict(all=list(range(len(rows))))
    for population in sorted({r['evaluation_population'] for r in rows}):
        selections[population] = [i for i,r in enumerate(rows) if r['evaluation_population']==population]
    if any(r['source']=='short_pulse' for r in rows):
        selections['planner_departure'] = [i for i,r in enumerate(rows) if r['source']=='short_pulse'
            and r['observation_horizon_receipt']['departure_tick']==13]
    for cluster in sorted({r['cluster'] for r in rows}):
        selections[cluster] = [i for i,r in enumerate(rows) if r['cluster']==cluster]
    zero = [0.,0.,0.]
    selections['future_steady'] = []; selections['future_switch'] = []; selections['future_brake'] = []
    for i,r in enumerate(rows):
        pairs = [(a,b) for a,b in zip(r['known_commands'],r['known_commands'][1:]) if a!=b]
        if not pairs: selections['future_steady'].append(i)
        else: selections['future_switch'].append(i)
        if any(a!=zero and b==zero for a,b in pairs): selections['future_brake'].append(i)
    roster = comparison_roster()
    OUTPUT.mkdir(); predictions = {}; bindings = {}
    try:
        for name,directory,record in roster:
            trainer = load(directory,record); model = trainer.model; model.eval()
            pred = []
            head = 'direct_outcomes' if trainer.condition=='direct' else 'rollout_outcomes'
            for start in range(0,len(rows),6):
                output = model(**stack_samples([cache[i] for i in range(start,min(start+6,len(rows)))]))
                raw = output[head].numpy()
                pred.append(np.concatenate((raw[:,:,:2],np.arctan2(raw[:,:,2:3],raw[:,:,3:4])),axis=-1))
            predictions[name] = np.concatenate(pred)
            bindings[name] = dict(directory=directory.name, filename=record['filename'], sha256=record['sha256'])
            print('PREDICTIONS_COMPLETE',name,len(rows),flush=True)
            del trainer, model
        nominal = np.zeros((len(rows),8,3))
        for i,row in enumerate(rows):
            state = np.zeros(3)
            for h,(vx,vy,w) in enumerate(row['known_commands']):
                angle = state[2]+.05*w
                state[:2] += .1*np.array([vx*np.cos(angle)-vy*np.sin(angle),vx*np.sin(angle)+vy*np.cos(angle)])
                state[2] += .1*w; nominal[i,h] = state
        predictions['command_integrated'] = nominal
        predictions['stationary_persistence'] = np.zeros_like(nominal)
        scores = {name:{group:summarize(pred,rows,ids) for group,ids in selections.items()}
                  for name,pred in predictions.items()}
        np.savez_compressed(OUTPUT/'predictions.npz',**predictions)
        (OUTPUT/'sample_ids.json').write_text(json.dumps([r['sample_id'] for r in rows]))
        report = dict(status='COMPLETE',available_windows=len(rows),scores=scores,model_bindings=bindings,
            consumed_policy_sha256=policy_hashes, wall_s=time.monotonic()-began,
            target_windows_sha256=hashlib.sha256((TARGETS/'windows.json').read_bytes()).hexdigest(),
            future_rgb_used_for_inference=False, inference_native_state_used=False,
            native_pose_is_evaluation_target_only=True, prediction_interpretation=PREDICTION_INTERPRETATION,
            external_fitted_residual_correction_applied=False, fixed_budget_final_checkpoints=True,
            independent_parameter_clusters=2, overlapping_contexts=True, independent_maze_trials=0,
            closed_loop_improvement_established=False)
        (OUTPUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:{g:v[g]['700'] for g in ('all','original','pre_switch','future_switch')}
                          for k,v in scores.items()},indent=2),flush=True)
    except Exception as error:
        (OUTPUT/'failure.json').write_text(json.dumps(dict(reason=repr(error))))
        raise


if __name__=='__main__': main()
