"""Bound predecessor witnesses and evaluation-only new-observation comparisons.

No native collection, model fitting, navigation promotion or dataset discovery.
Nonidentity is necessary evidence here, never statistical independence or maze
topology novelty. Packet labels and command changes cannot establish new views.
"""
from itertools import product
import hashlib
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts import navigation_artifact_root_development as custody
from scripts.independent_tracking_cohort_development import require, read
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

PRIOR_TRIALS = ('nominal_left','nominal_right','lower_friction_left')
PREFIX_FRAMES = 9
PREFIX_SAMPLES = 1150
GEOMETRY_NUMERIC_TOLERANCE_M = 1e-5
MINIMUM_START_SEPARATION_M = .05
MINIMUM_START_SEPARATION_RAD = .05
PREDECESSORS = {
    'inner':dict(root='go2_inner_arrival_room_return_v1_attempt_001',
        launch='53ec5a2a04831ddbc8ae3932038ce90a5f12c797d38087bb6fb95a3e8778667a',
        result='938090fce09f77b9c55c919d6ee3b29d71dde5b5939374a542aa0edf5a7c0456',
        audit='07ae6d3a3ff3183e015617159cbd499faf751cbf13af646ac1ddcfa2bc0b8810'),
    'intent':dict(root='go2_intent_room_return_v1_attempt_001',
        launch='7a5c427ca521de367a301376fafd262aeda5f6e16b7ed876f2b40e87ce0b1a91',
        result='27e2f91eaece8667e48fb98d75ce3ee8cfcc3a1c9d0052a97033cad5acc321d3',
        audit='a350c74d4f5851a7486bf01a8276be7f9eb420a83cd8198ad684159b8385923d')}


def geometry_witness(static):
    """World-space corners ignore names, materials, quaternion sign and axes."""
    require(type(static) is list and 1<=len(static)<=64, 'bounded nonempty actual native wall inventory')
    boxes=[]
    for row in static:
        require(row['fixed'] is True and row['collision_enabled'] is True
            and row['native_collision_boxes']==1, 'one fixed enabled native box per wall required')
        position=np.asarray(row['native_position'],float)
        size=np.asarray(row['native_box_size'],float)
        quat=np.asarray(row['native_quaternion_wxyz'],float)
        require(position.shape==(3,) and size.shape in ((3,),(7,)) and quat.shape==(4,)
            and all(np.isfinite(v).all() for v in (position,size,quat))
            and np.all(size[:3]>0) and np.all(size[3:]==0)
            and abs(np.linalg.norm(quat)-1)<1e-6, 'finite actual box geometry required')
        R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        corners=np.asarray(list(product((-.5,.5),repeat=3)))*size[:3]
        boxes.append((corners@R.T+position).tolist())
    return dict(native_box_corners_world_m=boxes, wall_count=len(boxes),
        labels_used=False, topology_isomorphism_test=False)


def _same_points(a,b,tolerance):
    distance=np.linalg.norm(a[:,None]-b[None,:],axis=2)
    # Minimize unmatched edges, not the sum of distances: the latter need not
    # find a within-tolerance perfect matching even when one exists.
    i,j=linear_sum_assignment(distance>tolerance)
    return bool(np.all(distance[i,j]<=tolerance))


def same_geometry(a,b):
    aa=np.asarray(a['native_box_corners_world_m'],float)
    bb=np.asarray(b['native_box_corners_world_m'],float)
    for data,witness in ((aa,a),(bb,b)):
        require(data.shape==(witness['wall_count'],8,3) and 1<=len(data)<=64 and np.isfinite(data).all(),
                'complete finite corner witness required')
    if len(aa)!=len(bb):return False
    match=np.array([[_same_points(x,y,GEOMETRY_NUMERIC_TOLERANCE_M) for y in bb] for x in aa])
    i,j=linear_sum_assignment(~match)
    return bool(match[i,j].all())


def sensor_value_identity(packet):
    """Separate sensory values from time, calibration labels, identity and control.

    The reader validates those contracts separately. An invalid sensor contract
    cannot be repaired merely by ignoring its metadata for this comparison.
    """
    policy,depth,fast,_=packet
    sensed=policy['sensor_state']['sensed']
    return dict(rgb=fingerprint(policy['image']['rgb']),
        depth=fingerprint((depth['depth_m'],depth['valid'])),
        body=fingerprint({k:(v['values'],v['valid']) for k,v in sensed.items()}),
        fast_gyro=fingerprint((fast['values'],fast['valid'])),
        control=fingerprint({k:(v['values'],v['valid']) for k,v in policy['sensor_state']['control'].items()}))


def extract_witness(raw,static,reader,cameras):
    """Evaluation-only arrays supplied after admission; missing prefixes stay missing."""
    times=np.asarray(raw['timestamp_s']);pose=np.asarray(raw['base_pose_world'])
    frames=len(reader.frames) if reader is not None else 0
    require(times.ndim==1 and pose.shape==(len(times),7) and np.isfinite(times).all()
        and np.isfinite(pose).all(), 'finite complete pose/time arrays required')
    require(np.allclose(times,np.arange(1,len(times)+1)*.002,rtol=0,atol=1e-10), 'fixed500Hz native clock required')
    require(len(cameras)==frames,'complete camera association population required')
    geometry=geometry_witness(static) if static else None
    initial=pose[0].tolist() if len(pose) else None
    settled=pose[749].tolist() if len(pose)>=750 else None
    common=dict(native_geometry=geometry,actual_initial_pose_world=initial,actual_settled_pose_world=settled,
        native_samples=len(times),frames=frames,evaluator_only=True,
        independent_observations_verified=False,navigation_qualified=False)
    if len(times)<PREFIX_SAMPLES or frames<PREFIX_FRAMES or geometry is None:
        return common|dict(status='MISSING_COMPLETE_GEOMETRY_START_OR_SENSOR_PREFIX',sensor_values=[],
                           native_pose_prefix_sha256=None)
    require(np.allclose(np.linalg.norm(pose[:PREFIX_SAMPLES,3:],axis=1),1,rtol=0,atol=1e-6),
            'unit native prefix quaternion required')
    values=[]
    for frame in range(PREFIX_FRAMES):
        packet=reader.packet(frame);now=1_500_000_000+frame*100_000_000;sample=749+50*frame
        require(packet[3]==now and cameras[frame]['physical_sample_index']==sample
            and abs(times[sample]*1e9-now)<1
            and cameras[frame]['rgb_sha256']==hashlib.sha256(packet[0]['image']['rgb'].tobytes()).hexdigest(),
            'actual native/RGB prefix frame and content association required')
        values.append(sensor_value_identity(packet))
    return common|dict(status='COMPLETE_VALUE_ONLY_PREFIX',sensor_values=values,
        native_pose_prefix_sha256=fingerprint(pose[:PREFIX_SAMPLES]),
        prefix_frames=PREFIX_FRAMES,prefix_native_samples=PREFIX_SAMPLES)


def compare_witnesses(reference,candidate):
    if reference['status']!='COMPLETE_VALUE_ONLY_PREFIX' or candidate['status']!='COMPLETE_VALUE_ONLY_PREFIX':
        return dict(status='UNAVAILABLE_COMPARISON',nonidentity_checks_pass=False,
            independent_observations_verified=False,navigation_qualified=False)
    same=same_geometry(reference['native_geometry'],candidate['native_geometry'])
    a=np.asarray(reference['actual_initial_pose_world'],float)
    b=np.asarray(candidate['actual_initial_pose_world'],float)
    require(a.shape==b.shape==(7,) and np.isfinite(a).all() and np.isfinite(b).all(), 'actual initial poses required')
    displacement=float(np.linalg.norm(a[:2]-b[:2]))
    rotation=float((Rotation.from_quat(a[3:]).inv()*Rotation.from_quat(b[3:])).magnitude())
    separated=displacement>=MINIMUM_START_SEPARATION_M or rotation>=MINIMUM_START_SEPARATION_RAD
    av,bv=reference['sensor_values'],candidate['sensor_values']
    require(len(av)==len(bv)==PREFIX_FRAMES, 'all nine value witnesses required')
    fields={'rgb','depth','body','fast_gyro','control'}
    require(all(set(v)==fields and all(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h)
        for h in v.values()) for v in av+bv), 'exact sensor-value SHA witnesses required')
    unequal={k:[i for i,(x,y) in enumerate(zip(av,bv,strict=True)) if x[k]!=y[k]] for k in sorted(fields)}
    # Require a different initial RGB AND depth view, not just subsequent motion
    # under different commands, labels, numerical poses or other sensor noise.
    new_initial_view=0 in unequal['rgb'] and 0 in unequal['depth']
    passed=not same and separated and new_initial_view
    return dict(status='COMPARED_ACTUAL_GEOMETRY_START_AND_SENSOR_VALUES',
        native_box_inventory_matches=same,initial_planar_separation_m=displacement,
        initial_rotation_separation_rad=rotation,initial_start_separated=separated,
        unequal_value_frames=unequal,initial_rgb_and_depth_values_differ=new_initial_view,
        native_pose_prefix_matches=reference['native_pose_prefix_sha256']==candidate['native_pose_prefix_sha256'],
        nonidentity_checks_pass=bool(passed),geometry_numeric_tolerance_m=GEOMETRY_NUMERIC_TOLERANCE_M,
        numerical_tolerance_is_physical_error_bound=False,command_difference_is_observational_novelty=False,
        rigid_transform_or_topology_isomorphism_test=False,independent_observations_verified=False,
        maze_generalization_established=False,navigation_qualified=False)


def _selected_names(trial,count):
    require(trial in PRIOR_TRIALS and type(count) is int and PREFIX_FRAMES<=count<=3611, 'bounded complete prior trial')
    fixed=('static_objects.json','physics_trace.npz','camera_audit.json','policy_observations.json',
           'policy_histories.npz','depth_observations.json','fast_gyro_histories.npz')
    return {trial+'/'+n for n in fixed}|{trial+f'/{kind}_{i:04d}.{suffix}'
        for i in range(PREFIX_FRAMES) for kind,suffix in (('rgb','png'),('depth','npz'))}


def _read_pose(output,name):
    # Only these numeric arrays are parsed. The entire NPZ byte binding is
    # checked first, but unrelated native fields are not materialized here.
    with np.load(custody.artifact_path(output,name),allow_pickle=False) as archive:
        return {k:archive[k] for k in ('timestamp_s','base_pose_world')}


def load_predecessors():
    """Read-only exact six previously exposed recordings; no recursive discovery."""
    cohorts={}
    for key,descriptor in PREDECESSORS.items():
        output=custody.validate_root(custody.BASE/descriptor['root'])
        receipts={'launch.json':descriptor['launch'],'result.json':descriptor['result'],
                  'raw_return_audit.json':descriptor['audit']}
        custody.verify_artifacts(output,receipts)
        launch=read(output,'launch.json');result=read(output,'result.json');audit=read(output,'raw_return_audit.json')
        require(launch['output_root']==str(output) and result['status']=='ROOM_RETURN_PULSE_COLLECTION_TERMINAL'
            and result['absent_expected_artifacts']==[] and set(result['conditions'])==set(PRIOR_TRIALS)
            and audit['status']=='RAW_RETURN_AUDIT_PASS' and set(audit['conditions'])==set(PRIOR_TRIALS)
            and result['goal_achieved'] is False and audit['goal_achieved'] is False,
            'exact completed predecessor collection/raw audit required; no success relabeling')
        selected={};witnesses={}
        for trial in PRIOR_TRIALS:
            r=result['conditions'][trial];a=audit['conditions'][trial]
            require(a['raw_sensor_audit_pass'] is True and a['physics_samples']==r['physics_samples']
                and a['raw_depth_checks']==r['rgbd_frames'], 'matching terminal raw sensor audit required')
            names=_selected_names(trial,r['rgbd_frames'])
            require(names<=set(result['artifact_sha256']), 'every accessed predecessor artifact must be bound')
            bindings={n:result['artifact_sha256'][n] for n in sorted(names)}
            custody.verify_artifacts(output,bindings);selected.update(bindings)
            reader=IntentReturnRGBDReplay(output/trial)
            require(len(reader.frames)==r['rgbd_frames'], 'complete predecessor frame population required')
            raw=_read_pose(output,trial+'/physics_trace.npz')
            require(len(raw['timestamp_s'])==r['physics_samples'], 'complete predecessor native population required')
            w=extract_witness(raw,read(output,trial+'/static_objects.json'),reader,read(output,trial+'/camera_audit.json'))
            require(w['status']=='COMPLETE_VALUE_ONLY_PREFIX','complete declared predecessor prefix required')
            witnesses[trial]=w|dict(original_full_room_return_success=a['full_room_return_success'])
        custody.verify_artifacts(output,receipts|selected)
        cohorts[key]=dict(root=str(output),receipt_sha256=receipts,selected_artifact_sha256=selected,witnesses=witnesses)
    return dict(schema='independent_tracking_predecessor_value_witnesses.v1',cohorts=cohorts,
        prefix_frames=PREFIX_FRAMES,prefix_native_samples=PREFIX_SAMPLES,
        new_native_collection_performed=False,new_observer_replay_performed=False,
        independent_observations_verified=False,navigation_qualified=False)


def compare_new_population(output,collection_sha256,base_phase_sha256,stress_phase_sha256,result_sha256):
    """Evaluation only: new native bytes cannot be parsed before all sensor phases.

    Returns a report to a future frozen launcher; does not create/overwrite a
    result artifact or alter the existing base/stress result's qualification.
    """
    from scripts.independent_tracking_stress_cohort_development import admit_complete_sensor_phase
    from lewm.independent_tracking_challenge_development import TRIALS
    episodes,_,_=admit_complete_sensor_phase(output,collection_sha256,base_phase_sha256,stress_phase_sha256)
    custody.verify_artifacts(output,{'result.json':result_sha256});result=read(output,'result.json')
    require(result['status']=='EIGHT_TRIAL_BASE_AND_FIXED_STRESS_RAW_AUDIT_AND_SCORING_COMPLETE'
        and result['collection_sha256']==collection_sha256 and result['base_phase_sha256']==base_phase_sha256
        and result['stress_phase_sha256']==stress_phase_sha256 and result['trials']==list(TRIALS),
        'complete bound raw audit and base/stress score result required before comparison')
    required={t+'_audit.json' for t in TRIALS}
    require(required<=set(result['output_sha256']), 'all actual raw-audit results required')
    custody.verify_artifacts(output,result['output_sha256'])
    predecessors=load_predecessors();comparisons={};current={}
    for trial in TRIALS:
        entry=episodes[trial]['result'];directory=output/trial
        raw=_read_pose(output,trial+'/physics_trace.npz')
        reader=IntentReturnRGBDReplay(directory) if entry['rgbd_frames'] else None
        static=read(output,trial+'/static_objects.json') if entry['setup_checked'] else []
        witness=extract_witness(raw,static,reader,read(output,trial+'/camera_audit.json'));current[trial]=witness
        comparisons[trial]={key+'/'+old:compare_witnesses(previous,witness)
            for key,cohort in predecessors['cohorts'].items() for old,previous in cohort['witnesses'].items()}
    admit_complete_sensor_phase(output,collection_sha256,base_phase_sha256,stress_phase_sha256)
    custody.verify_artifacts(output,{'result.json':result_sha256}|result['output_sha256'])
    return dict(status='ACTUAL_PREDECESSOR_NONIDENTITY_COMPARISON_COMPLETE',result_sha256=result_sha256,
        predecessors=predecessors,current_witnesses=current,comparisons=comparisons,
        all_six_predecessor_nonidentity_checks_pass=all(
            c['nonidentity_checks_pass'] for row in comparisons.values() for c in row.values()),
        independent_observations_verified=False,full_challenge_pass=False,
        navigation_qualified=False,goal_achieved=False)
