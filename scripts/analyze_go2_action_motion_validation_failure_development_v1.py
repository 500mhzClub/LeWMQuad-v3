"""Read-only frozen-B fusion reconstruction and evaluator-only descriptive errors.

No failed controller is resumed. A fresh offline sensor integrator is replayed
only through the recorded terminal decision, never into the stopping tail.
"""
import json

import numpy as np

from lewm.action_motion_identification_development import motion_priors
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.setup_velocity_prior_development import SetupVelocityIntegrator
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_action_motion_validation_development_v1 import OUTPUT, PROTOCOL
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_startup_observation_turn_development_v1 import verify_extensions
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json

IDENTITIES={
    'launch.json':'5f6131be108b1a7ad118108eca7ebecb1dccb5e15daf85e6f391d5c321addae2',
    'result.json':'9c2c7810f511e052466368fddf5833d84abb945a2a34db166a64fbc6e3714bbc',
    'raw_artifact_audit.json':'ba9504f5c3c001a3e0f5c55f45f9f847e0f4d1c97806fc3b7041734dd68b63f9'}
TARGET=ROOT/'docs/go2_action_motion_validation_development_v1_diagnostic_2026-09-06.json'


def analyze():
    bindings={str((OUTPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}
    verify_bindings(bindings)
    launch=read_json(OUTPUT,'launch.json'); result=read_json(OUTPUT,'result.json')
    audit=read_json(OUTPUT,'raw_artifact_audit.json')
    bindings|=launch['source_sha256']|launch['input_sha256']|{
        str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    own='scripts/analyze_go2_action_motion_validation_failure_development_v1.py'
    bindings[own]=digest(ROOT/own)
    def verify():
        verify_bindings(bindings);verify_native_bindings(launch['native_sha256']);verify_extensions(launch['native_geometry_sha256'])
    verify()
    decisions=read_json(OUTPUT,'motion_decisions.json'); relatives=read_json(OUTPUT,'relative_state_observations.json')
    cameras=read_json(OUTPUT,'camera_audit.json')
    prior,_=motion_priors(1_500_000_000,launch['source_sha256'][PROTOCOL]); integrator=SetupVelocityIntegrator(prior)
    rows=[]
    with np.load(OUTPUT/'physics_trace.npz',allow_pickle=False) as raw:
        anchor=raw['base_pose_world'][cameras[0]['physical_sample_index']]; R0=rotation_xyzw(anchor[3:])
        for frame,item in enumerate(decisions):
            policy,_=load_rgbd_observation(OUTPUT,frame)
            relative=relatives[frame]['observer']; fused=integrator.observe(policy,relative)
            recorded=item['decision']['state']; motion=relative['motion']
            if not recorded['terminal']:
                assert np.isclose(fused['position_error_scale_m'],recorded['combined_position_scale_m'],atol=1e-12,rtol=0)
                assert fused['usable_under_declared_proxy_budget']
            else:
                assert frame==len(decisions)-1 and not fused['usable_under_declared_proxy_budget']
            index=cameras[frame]['physical_sample_index']; actual=R0.T@(raw['base_pose_world'][index,:3]-anchor[:3])
            rows.append(dict(measured_ns=fused['measured_ns'],depth_rank=fused['depth_rank'],
                weak_directions_previous_body=motion['weak_directions_previous_body'] if motion else [],
                consecutive_weak_seconds=fused['consecutive_weak_seconds'],
                inherited_position_error_scale_m=fused['inherited_position_error_scale_m'],
                initial_velocity_radius_contribution_m=fused['initial_velocity_prior_transport']['position_radius_m'],
                combined_position_scale_m=fused['position_error_scale_m'],
                declared_budget_m=fused['assumptions']['maximum_position_scale_m'],
                usable_under_declared_proxy_budget=fused['usable_under_declared_proxy_budget'],
                estimated_position_initial_body_m=fused['position_initial_body_m'],
                evaluation_only_position_error_m=float(np.linalg.norm(np.asarray(fused['position_initial_body_m'])-actual)),
                terminal_controller=recorded['terminal']))
        last_decision=cameras[decisions[-1]['observation_index']]['physical_sample_index']
        tail_displacement=float(np.linalg.norm(raw['base_pose_world'][-1,:3]-raw['base_pose_world'][last_decision,:3]))
    timings=read_json(OUTPUT,'motion_timings.json')
    active=[r['outer_wall_ms'] for r in timings['outer'] if r['fresh_capture_inside_loop']
            and r['command_tick_attempted'] and r['completed_without_exception']]
    profile={name:[] for name in ('capture','controller','execution','other')}
    tape=read_json(OUTPUT,'motion_command_tape.json')
    for row in timings['outer']:
        if not row['fresh_capture_inside_loop'] or not row['command_tick_attempted'] or not row['completed_without_exception']:continue
        i=row['decision_index']; frame=row['observation_index']
        capture=timings['captures'][frame]['acquisition_and_depth_observer_ms']
        controller=timings['controller'][i]['controller_wall_ms']
        execution=next(r['execution_wall_ms'] for r in tape if r['decision_index']==i)
        for key,value in zip(profile,(capture,controller,execution,row['outer_wall_ms']-capture-controller-execution),strict=True):
            profile[key].append(value)
    verify()
    return dict(status='B_TERMINAL_FUSION_RECONSTRUCTED_NO_RESUME',B_identities=IDENTITIES,
        diagnostic_source_sha256={own:bindings[own]},fusion_rows=rows,
        three_tick_tail_actual_displacement_m=tail_displacement,
        active_full_loop_wall_ms=dict(count=len(active),minimum=min(active),median=float(np.median(active)),
            maximum=max(active),above_100ms=sum(v>100 for v in active)),
        mean_loop_components_ms={k:float(np.mean(v)) for k,v in profile.items()},
        model_summaries={k:v['summary'] for k,v in audit['prediction_errors'].items()},
        native_truth_enters_sensor_estimation=False,calibrated_error_bound=False,
        B_retried=False,failed_controller_resumed=False,navigation_qualified=False)


def main():
    if TARGET.exists(): raise ValueError('new diagnostic output only; no overwrite')
    report=analyze();write_json(TARGET,report)
    print(json.dumps(dict(terminal=report['fusion_rows'][-1],timing=report['active_full_loop_wall_ms'],
        profile=report['mean_loop_components_ms'],tail_m=report['three_tick_tail_actual_displacement_m']),allow_nan=False),flush=True)


if __name__=='__main__':main()
