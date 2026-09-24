"""Distinct post-hoc raw audit of the failed tracking tapes.

Narrow source adaptation of independent_tracking_evaluation_development._raw_audit:
only the function name/docstring and coverage callee/import differ. Every raw
sensor/contact/setup/command/stop/raster algorithm and check is retained. The
coverage field now contains explicit representation provenance and nested coverage.
No old source or output is changed. Admission is the caller's responsibility.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

from lewm.independent_tracking_challenge_development import specification
from lewm.representation_aware_tracking_coverage_development import coverage_with_sensor_rotation_convention
from scripts.independent_tracking_cohort_development import require,read,IntentReturnRGBDReplay
from scripts.independent_tracking_command_audit_development import audit_commands
from scripts.near_field_sensor_audit_development import audit_sensors,read_npz
from scripts.audit_go2_independent_pulse_context_pilot_v1 import audit_setup,audit_stops,prefix_witness
from lewm.raster_footprint_visibility_development import evaluate_footprint
from scripts.independent_tracking_native_contact_guard_development import validate_report as validate_native_contact_report

def raw_audit_sensor_convention(output,trial,result,protocol_sha256):
    """Caller must authenticate all 96 admitted streams before this boundary."""
    directory=output/trial;spec=specification(trial)
    raw,contacts,topology,roles,cameras,_,geometry,sensors=audit_sensors(directory,spec,result)
    n=len(raw['timestamp_s'])
    native_contact_integrity=validate_native_contact_report(result['native_contact_integrity'],n)
    np.testing.assert_allclose(raw['base_pose_world'][0,:2],spec['geometry']['spawn_se2_world'][:2],atol=.002,rtol=0)
    yaw=Rotation.from_quat(raw['base_pose_world'][0,3:]).as_euler('xyz')[2]
    delta=yaw-spec['geometry']['spawn_se2_world'][2]
    require(abs(math.atan2(math.sin(delta),math.cos(delta)))<.002,'actual initial heading differs from construction')
    friction=read(output,trial+'/friction_checks.json')
    require(friction[0]['stage']=='before_settle' and friction[0]['physics_steps']==0
            and friction[-1]['stage']=='terminal' and friction[-1]['physics_steps']==n,
            'friction witnesses must cover actual construction/termination')
    for f in friction:
        np.testing.assert_allclose(f['solver_friction'],spec['friction_mu'],atol=1e-7,rtol=0)
        np.testing.assert_array_equal(f['solver_ratio'],np.ones((1,28)))
    require(read(output,trial+'/actuator_identity.json')['effective']==read(output,trial+'/terminal_actuator_gains.json'),
            'frozen native gait gains changed')
    rows=read(output,trial+'/tracking_decisions.json');tape=read(output,trial+'/command_tape.json')
    require(len(rows)<=len(friction)-2<=len(rows)+1,'complete pre-decision friction population required')
    for tick,f in enumerate(friction[1:-1]):
        require(f['stage']=='before_decision' and f['tick']==tick and f['physics_steps']==750+50*tick,
                'friction witness/actual command clock mismatch')
    command=audit_commands(raw,tape,rows,result,spec['direction'])
    from lewm.independent_tracking_challenge_development import decision
    reader=IntentReturnRGBDReplay(directory) if cameras else None
    for tick,row in enumerate(rows):
        p,_d,_f,now=reader.packet(tick)
        require(decision(spec['direction'],tick,p)==row['decision'] and now==row['decision']['decision_ns'],
                'saved command differs from real causal-packet selector')
    setup=audit_setup(directory,raw,contacts,topology,geometry,result,protocol_sha256)
    stop=audit_stops(raw,contacts,roles,friction,setup,read(output,trial+'/native_guard_rows.json'),result)
    rasters=[];footprints=[]
    for i,camera in enumerate(cameras):
        raster=read(output,trial+f'/raster_{i:04d}.json')
        require(raster['physical_sample_index']==camera['physical_sample_index']
                and raster['order']['order']=='floor_first' and raster['order']['roles']==['floor','walls']
                and set(raster['order']['surfaces'])=={'floor','walls'},'unchanged actual core raster ordering required')
        p=raster['precision'];positions=p['rgb_target_sample_positions']
        require(1<=p['subpixel_bits']<=32 and 1<=p['depth_target_depth_bits']<=64
                and 1<=p['rgb_target_samples']<=32 and len(positions)==p['rgb_target_samples']
                and all(len(v)==2 and all(0<=x<=1 for x in v) for v in positions),'actual raster precision contract')
        if rasters:require(raster['order']==rasters[0]['order'] and p==rasters[0]['precision'],'raster drift')
        rasters.append(raster)
        depth=read_npz(directory,f'native_depth_{i:04d}.npz')['optical_depth_m']
        score=evaluate_footprint(depth,spec['geometry']['wall_boxes'],camera['world_from_optical'],render_near_m=.005)
        require(score['original_strict_score']==sensors['depth_checks'][i]['physical_visibility'],
                'unchanged strict visibility score required')
        footprints.append(dict(frame=i,score=score))
    coverage=coverage_with_sensor_rotation_convention(spec['direction'],raw['timestamp_s'],raw['base_pose_world'],raw['base_twist_world'],
        completed_ticks=result['completed_ticks'],schedule_complete=result['schedule_complete'],
        physical_stop=result['physical_stop'],acquisition_stop=result['acquisition_stop'])
    return raw,dict(command=command,setup=setup,first_physical_stop=stop,sensors=sensors,
        native_contact_integrity=native_contact_integrity,
        coverage=coverage,prefix=prefix_witness(raw,contacts,reader),raster_readbacks=rasters,footprints=footprints,
        actual_initial_pose_world=raw['base_pose_world'][0].tolist(),
        actual_settled_pose_world=raw['base_pose_world'][749].tolist() if n>=750 else None,
        predecessor_prefix_comparison_performed=False,independent_observations_verified=False,navigation_qualified=False)

