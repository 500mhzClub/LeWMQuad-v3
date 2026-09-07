"""Post-sensor-phase raw audit and paired pose/coverage scoring for eight tapes.

No training, native collection or controller promotion. Baseline challenge only:
stress-arm and predecessor-prefix independence evidence remain separate gates.
"""
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation

from lewm.independent_tracking_challenge_development import TRIALS,specification
from lewm.independent_tracking_coverage_development import measured_coverage
from scripts.independent_tracking_cohort_development import (
    ARMS,MAX_ROW,require,read,admit_native_evaluation,verify_collection,IntentReturnRGBDReplay,record_phase_failure)
from scripts.independent_tracking_command_audit_development import audit_commands
from scripts.near_field_sensor_audit_development import audit_sensors,read_npz
from scripts.audit_go2_independent_pulse_context_pilot_v1 import audit_setup,audit_stops,prefix_witness
from scripts.verify_go2_temporal_anchor_continuity_v1 import pose_errors,statistics
from scripts.navigation_artifact_root_development import verify_artifacts,artifact_path
from lewm.raster_footprint_visibility_development import evaluate_footprint
from scripts.independent_tracking_native_contact_guard_development import validate_report as validate_native_contact_report

POSITION_ALLOCATION_M=.02
ORIENTATION_ALLOCATION_RAD=math.radians(2)


def _raw_audit(output,trial,result,protocol_sha256):
    """Internal only: public evaluate_population authenticates all sensor rows first."""
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
    coverage=measured_coverage(spec['direction'],raw['timestamp_s'],raw['base_pose_world'],raw['base_twist_world'],
        completed_ticks=result['completed_ticks'],schedule_complete=result['schedule_complete'],
        physical_stop=result['physical_stop'],acquisition_stop=result['acquisition_stop'])
    return raw,dict(command=command,setup=setup,first_physical_stop=stop,sensors=sensors,
        native_contact_integrity=native_contact_integrity,
        coverage=coverage,prefix=prefix_witness(raw,contacts,reader),raster_readbacks=rasters,footprints=footprints,
        actual_initial_pose_world=raw['base_pose_world'][0].tolist(),
        actual_settled_pose_world=raw['base_pose_world'][749].tolist() if n>=750 else None,
        predecessor_prefix_comparison_performed=False,independent_observations_verified=False,navigation_qualified=False)


def _score_trial(store,trial,report,raw):
    frames=report['frames'];native=raw['base_pose_world'];times=raw['timestamp_s']
    require(native.shape==(len(times),7) and np.isfinite(native).all(),'complete finite native pose array required')
    metrics=('position_m','orientation_rad','incremental_position_m','incremental_orientation_rad')
    values={a:{k:[] for k in (*metrics,'paired_position_m','paired_orientation_rad','observer_wall_ms')} for a in ARMS}
    bridge={k:[] for k in metrics};previous={a:None for a in ARMS};count=0
    if frames:
        require(len(native)>=750,'native initial body frame required')
        R0=Rotation.from_quat(native[749,3:]).as_matrix()
    with artifact_path(store.output,report['estimates_file']).open('rb') as source,store.stream(trial+'_evaluation.jsonl') as target:
        for line in source:
            require(len(line)<=MAX_ROW,'bounded paired input row')
            row=json.loads(line);sample=749+50*count
            require(row['frame']==count and sample<len(times) and abs(times[sample]*1e9-row['measured_ns'])<1.,
                    'native/sensor frame and clock association required')
            truth_p=(native[sample,:3]-native[749,:3])@R0
            truth_R=R0.T@Rotation.from_quat(native[sample,3:]).as_matrix()
            errors={};both=row['availability']=='both'
            for arm in ARMS:
                r=row['arms'][arm];v=values[arm];v['observer_wall_ms'].append(r['observer_wall_ms'])
                if r['pose'] is None:
                    errors[arm]=None;previous[arm]=None;continue
                p=np.asarray(r['pose']['position_initial_body_m']);R=np.asarray(r['pose']['rotation_initial_body_from_current_body'])
                e=pose_errors(p,R,truth_p,truth_R,previous[arm]);errors[arm]=e
                previous[arm]=(p,R,truth_p,truth_R)
                for k in metrics:
                    if e[k] is not None:v[k].append(e[k])
                if both:
                    v['paired_position_m'].append(e['position_m']);v['paired_orientation_rad'].append(e['orientation_rad'])
                if arm=='temporal_anchor' and r['continuity']['status']=='MEASURED_INCREMENT_BRIDGE':
                    require(all(e[k] is not None for k in metrics),'bridge requires consecutive measured estimates')
                    for k in metrics:bridge[k].append(e[k])
            store.append(target,dict(frame=count,measured_ns=row['measured_ns'],native_sample_index=sample,
                availability=row['availability'],errors=errors,evaluator_only=True,navigation_qualified=False));count+=1
    require(count==frames and all(len(values[a]['position_m'])==report['arms'][a]['available'] for a in ARMS),
            'complete availability-conditioned scoring denominators required')
    require(len(bridge['position_m'])==report['continuity']['total_bridged_frames'],'complete bridge scoring population required')
    stats={a:{k:statistics(v) for k,v in fields.items()} for a,fields in values.items()}
    within={a:bool(frames>0 and stats[a]['position_m']['count']==frames
        and stats[a]['position_m']['maximum']<=POSITION_ALLOCATION_M
        and stats[a]['orientation_rad']['maximum']<=ORIENTATION_ALLOCATION_RAD) for a in ARMS}
    return dict(frames=frames,arms=stats,bridged_frame_errors={k:statistics(v) for k,v in bridge.items()},
        empirical_local_pose_allocation_met=within,availability=report['availability'],
        empirical_allocation_is_calibrated_bound=False,incremental_error_is_consecutive_estimate_difference=True,
        observer_timing_excludes_acquisition_and_control=True,navigation_qualified=False)


def _evaluate_population(store,collection_sha256,sensor_phase_sha256,protocol_sha256):
    require(not store.failed,'failed cohort cannot enter native evaluation')
    episodes,phase=admit_native_evaluation(store.output,collection_sha256,sensor_phase_sha256)
    require(tuple(store.episodes)==TRIALS and store.episodes==episodes,'same admitted complete cohort required')
    require(type(protocol_sha256) is str and len(protocol_sha256)==64
            and all(c in '0123456789abcdef' for c in protocol_sha256),'exact setup protocol identity required')
    audits={};scores={}
    for trial in TRIALS:
        raw,audit=_raw_audit(store.output,trial,episodes[trial]['result'],protocol_sha256)
        score=_score_trial(store,trial,phase['reports'][trial],raw)
        audits[trial]=audit;scores[trial]=score
        store.save(trial+'_audit.json',dict(raw_audit=audit,pose_score=score))
    # Reauthenticate all recorded bytes and sensor-only outputs after scoring.
    verify_collection(store.output,collection_sha256)
    verify_artifacts(store.output,{r['estimates_file']:r['estimates_sha256'] for r in phase['reports'].values()})
    result=dict(status='EIGHT_TRIAL_BASE_TRACKING_RAW_AUDIT_AND_SCORING_COMPLETE',collection_sha256=collection_sha256,
        sensor_phase_sha256=sensor_phase_sha256,protocol_sha256=protocol_sha256,
        trials=list(TRIALS),scores=scores,output_sha256=dict(store.hashes),
        all_intended_motion_covered=all(a['coverage']['intended_motion_covered'] for a in audits.values()),
        all_candidate_pose_allocations_met=all(s['empirical_local_pose_allocation_met']['temporal_anchor'] for s in scores.values()),
        strict_depth_visibility_pass=all(a['sensors']['depth_checks'] and all(
            x['within1mm'] and x['physical_visibility']['passes_sampled_physical_visibility']
            for x in a['sensors']['depth_checks']) for a in audits.values()),
        predecessor_prefix_comparison_performed=False,stress_arms_evaluated=False,
        independent_observations_verified=False,full_challenge_pass=False,
        navigation_qualified=False,real_time_qualified=False,goal_achieved=False)
    store.save('result.json',result)
    return result


def evaluate_population(store,collection_sha256,sensor_phase_sha256,protocol_sha256):
    require(not store.failed,'failed cohort cannot restart native evaluation')
    try:return _evaluate_population(store,collection_sha256,sensor_phase_sha256,protocol_sha256)
    except BaseException as error:
        record_phase_failure(store,'native_audit_or_admission',error)
        raise
