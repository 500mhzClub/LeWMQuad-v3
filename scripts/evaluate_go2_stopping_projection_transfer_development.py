"""Evaluate a completed transfer owner and check the actual controller treatment."""
import argparse
import hashlib
import json
import numpy as np

from lewm.eligible_floor_registration_development import bind
from lewm.commanded_planar_motion_development import forecast
from lewm.matched_motion_residual_runtime_development import FITS
from lewm.pose_command_xy_control_development import FIT_SHA256
from scripts.compare_continuous_navigation_arms_development import path, read, summarize
from scripts.evaluate_continuous_native_arrivals_development import evaluate as arrivals
from scripts.evaluate_saved_executed_motion_forecasts_development import evaluate as xy
from scripts.evaluate_saved_neural_command_yaw_development import evaluate as yaw
from scripts import run_go2_stopping_projection_transfer_development as cohort


def save_or_read(root, name, value):
    target=root/name
    if target.exists():
        saved=json.loads(target.read_text())
        if saved!=value:raise ValueError(f'preserve differing existing evaluation: {name}')
        return saved
    with target.open('x') as stream:json.dump(value,stream,indent=2)
    return value


def verify_treatment(root, condition):
    plans=[p for p in read(root,'planning.json') if 'selection' in p]
    expected_yaw='learned' if condition=='learned' else 'command'
    for p in plans:
        c=p.get('motion_correction') if condition=='reactive' else p['motion_correction']
        selection=p['selection']
        if condition=='reactive':
            if (c is not None or selection.get('candidate_future_outcomes_evaluated') is not False
                    or selection.get('learned_model_used') is not False
                    or 'planned_stopping_projection' in selection):
                raise ValueError('reactive action consumed predicted outcomes')
            continue
        expected_fit=FITS['supervised_rollout'][1] if condition=='learned' else FIT_SHA256
        if (c['forecast_xy_source']!=condition or c['forecast_yaw_source']!=expected_yaw
                or c['fit_sha256']!=expected_fit or c['contact_score_mode']!='disabled'
                or c['learned_yaw_retained']!=(condition=='learned')):
            raise ValueError('actual motion-source assignment differs')
        actual=np.asarray(c['applied_prediction_after_yaw_ablation'])
        upstream=np.asarray(c['upstream_prediction_for_yaw_ablation'])
        expected_xy=c['learned_corrected_forecast_xy_m' if condition=='learned' else 'pose_command_forecast_xy_m']
        command=forecast(p['committed_prefix'],pulse=bool(c['terminal_translation_pulse']))
        expected_rotation=upstream[:,:,2:4] if condition=='learned' else command[:,:,2:4]
        if (actual.shape!=(6,8,5) or not np.isfinite(actual).all()
                or not np.array_equal(actual[:,:,:2],c['corrected_forecast_xy_m'])
                or not np.allclose(actual[:,:,:2],expected_xy,rtol=0,atol=1e-7)
                or not np.allclose(actual[:,:,2:4],expected_rotation,rtol=0,atol=1e-7)
                or not np.all(actual[:,:,4]==-1000)
                or 'planned_stopping_projection' not in selection):
            raise ValueError('actual final scored forecast differs')
    return dict(condition=condition,selected_plans=len(plans),actual_treatment_verified=bool(plans),
        actual_xy_source='none' if condition=='reactive' else condition,
        actual_yaw_source='none' if condition=='reactive' else expected_yaw,
        predictive_outcomes_used=condition!='reactive')


def corrected_launch(root, condition, evidence):
    launch=read(root,'launch.json')
    if launch['comparison_condition']==condition:
        return launch
    if (not evidence['actual_treatment_verified'] or
            (launch['layout_index'],condition) not in ((0,'learned'),(1,'pose_command'))
            or launch['comparison_condition']!='supervised_rollout'):
        raise ValueError('unexpected annotation mismatch')
    captured=[]
    bind(cohort.annotate,TREATMENT=condition,RAW_WRITE=lambda name,value:captured.append(value))('launch.json',launch)
    intended=captured[0]
    fields={k:v for k,v in intended.items()
        if k not in ('extra_sources','source_sha256','owner') and launch.get(k)!=v}
    correction=dict(original_launch_sha256=hashlib.sha256((root/'launch.json').read_bytes()).hexdigest(),
        reason='cohort model-training CONDITION binding overwrote treatment annotation',
        corrected_fields=fields,actual_treatment_evidence=evidence,
        original_launch_preserved=True,runtime_and_recorded_forecasts_unchanged=True,
        source_exception='.generated/stopping_projection_transfer_annotation_fix_2026-09-15/correction.json')
    save_or_read(root,'launch_annotation_correction.json',correction)
    return launch|fields


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layout-index',type=int,choices=range(4),required=True)
    parser.add_argument('--condition',choices=cohort.CONDITIONS,required=True)
    args=parser.parse_args();root=path(cohort.ROOT.format(index=args.layout_index,condition=args.condition))
    evidence=verify_treatment(root,args.condition)
    save_or_read(root,'actual_controller_treatment_v1.json',evidence)
    launch=corrected_launch(root,args.condition,evidence)
    def evaluation_read(r,name):
        return launch if r==root and name=='launch.json' else read(r,name)
    physical=save_or_read(root,'continuous_native_arrival_evaluation.json',arrivals(root))
    summary=save_or_read(root,'live_navigation_summary_v1.json',summarize(root))
    if args.condition!='reactive':
        save_or_read(root,'saved_executed_motion_forecast_evaluation_v1.json',xy(root))
        yaw_result=bind(yaw,read=evaluation_read)(root)
        if (root/'launch_annotation_correction.json').exists():
            yaw_result['assignment_metadata_correction']='launch_annotation_correction.json'
        save_or_read(root,'saved_neural_command_yaw_evaluation_v1.json',yaw_result)
    print(json.dumps(dict(root_name=root.name,treatment=evidence,
        arrivals=physical['arrivals'],round_trip=physical['round_trip_arrival_checks_passed'],
        contacts=physical['disallowed_contact_samples'],poses=physical['registered_poses'],
        maximum_pose_error_m=physical['maximum_position_error_m'],
        terminal=physical['mission_terminal'],failure=summary['failure'])),flush=True)


if __name__=='__main__':main()
