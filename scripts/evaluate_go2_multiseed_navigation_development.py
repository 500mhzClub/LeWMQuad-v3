"""Evaluate every completed assignment, including failures, without changing data."""
import argparse
import json
from lewm.eligible_floor_registration_development import bind
from lewm.seeded_motion_correction_development import registry
from scripts import evaluate_go2_stopping_projection_transfer_development as previous
from scripts import run_go2_multiseed_navigation_development as study


def evaluate(index, arm):
    root = previous.path(study.ROOT.format(index=index, arm=arm))
    launch = previous.read(root, 'launch.json')
    assignment = study.assignment_for(arm)
    condition = arm if arm in ('reactive','pose_command') else 'learned'
    if (launch['model_assignment'] != assignment or launch['study_arm'] != arm
            or launch['comparison_condition'] != condition):
        raise ValueError('launch does not match the fixed assignment')
    variant = 'no_rgb' if '_no_rgb_' in assignment else 'full'
    entry = None if arm == 'reactive' else registry(variant)[assignment]
    fits = previous.FITS if entry is None else dict(supervised_rollout=(entry['root_name'],entry['fit_sha256']))
    evidence = bind(previous.verify_treatment, FITS=fits)(root, condition)
    for p in previous.read(root, 'planning.json'):
        if entry is None or 'selection' not in p: continue
        correction = p['motion_correction']['learned_motion_correction']
        if (correction['correction_base_model'] != assignment
                or correction['fit_sha256'] != entry['fit_sha256']
                or correction['correction_root'] != entry['root_name']):
            raise ValueError('recorded model-to-correction binding differs')
    evidence |= dict(study_arm=arm, model_assignment=assignment,
        model_and_correction_binding_verified=entry is not None and evidence['selected_plans'] > 0)
    previous.save_or_read(root,'actual_controller_treatment_v1.json',evidence)
    physical = previous.save_or_read(root,'continuous_native_arrival_evaluation.json',previous.arrivals(root))
    summary = previous.save_or_read(root,'live_navigation_summary_v1.json',previous.summarize(root))
    if entry is not None:
        previous.save_or_read(root,'saved_executed_motion_forecast_evaluation_v1.json',previous.xy(root))
        previous.save_or_read(root,'saved_neural_command_yaw_evaluation_v1.json',previous.yaw(root))
    result = dict(root_name=root.name, treatment=evidence, arrivals=physical['arrivals'],
        round_trip=physical['round_trip_arrival_checks_passed'], contacts=physical['disallowed_contact_samples'],
        poses=physical['registered_poses'], maximum_pose_error_m=physical['maximum_position_error_m'],
        terminal=physical['mission_terminal'], failure=summary['failure'])
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layout-index',type=int,choices=range(2),required=True)
    parser.add_argument('--arm',choices=study.ARMS,required=True)
    args=parser.parse_args()
    evaluate(args.layout_index,args.arm)


if __name__=='__main__': main()
