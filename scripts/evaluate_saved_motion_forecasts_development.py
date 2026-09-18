"""Evaluate saved selected forecasts only where actual commands match their prefix."""
import argparse
import json
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.physical_execution_development import rotation_xyzw
from scripts.navigation_artifact_root_development import BASE,validate_root


def metrics(errors):
    distance=np.linalg.norm(np.asarray(errors).reshape(-1,2),axis=1)
    if not len(distance):return dict(count=0)
    return dict(count=len(distance),rmse_m=float(np.sqrt(np.mean(distance**2))),
        median_m=float(np.median(distance)),p95_m=float(np.percentile(distance,95)),
        maximum_m=float(distance.max()))


def evaluate(root):
    validate_root(root,must_exist=True)
    read=lambda name:json.loads((root/name).read_text())
    frames={r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
    requests={r['now_ns']:r['requested_command'] for r in read('requests.json')}
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as native:
        poses=native['base_pose_world'].copy()
    groups={g:{str(h):dict(raw=[],corrected=[]) for h in (300,700,800)}
        for g in ('all','moving','translation','turn_only')}
    retained=[]
    for row in read('planning.json'):
        if 'motion_correction' not in row:continue
        frame=row['frame'];now=row['measured_ns'];index=ACTIONS.index(row['action'])
        command=candidate_commands(row['action'])[0]
        commands=np.asarray(row['committed_prefix']+[command]*4+[[0.,0.,0.]])
        assert commands.shape==(8,3)
        matches=[all(ns in requests and np.allclose(requests[ns],c,atol=1e-8,rtol=0)
            for ns in range(now+i*100_000_000,now+(i+1)*100_000_000,20_000_000))
            for i,c in enumerate(commands)]
        valid=np.logical_and.accumulate(matches)
        origin=poses[frames[frame]['physical_sample_index']]
        rotation=rotation_xyzw(origin[3:])
        moving=bool(np.any(commands[:7]));translation=bool(np.any(commands[:7,:2]))
        member=dict(all=True,moving=moving,translation=translation,turn_only=moving and not translation)
        horizons=[]
        for h in (3,7,8):
            if not valid[h-1] or frame+h not in frames:continue
            target=(rotation.T@(poses[frames[frame+h]['physical_sample_index'],:3]-origin[:3]))[:2]
            errors={name:np.asarray(row['motion_correction'][key])[index,h-1]-target
                for name,key in [('raw','raw_forecast_xy_m'),('corrected','corrected_forecast_xy_m')]}
            for group,include in member.items():
                if include:
                    for name,error in errors.items():groups[group][str(h*100)][name].append(error)
            horizons.append(h*100)
        if horizons:retained.append(dict(frame=frame,matched_horizons_ms=horizons))
    return dict(native_state_evaluator_only=True,online_forecasts_reused_without_recomputation=True,
        on_time_plans_only=False,
        actual_requested_command_prefix_required=True,counterfactual_outcomes_evaluated=False,
        model_refit=False,selected_actions_only=True,
        navigation_success_inferred_from_prediction_error=False,
        results={g:{h:{name:metrics(errors) for name,errors in values.items()}
            for h,values in horizons.items()} for g,horizons in groups.items()},
        retained_windows=retained)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    args=parser.parse_args();root=BASE/args.root_name
    report=evaluate(root)
    with (root/'saved_motion_forecast_native_evaluation.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report['results']))
