"""Fixed 144-cell moving-prefix/action-suffix development population."""
import random
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.geometry_progress_layout_family_development import assignments as old_assignments, specification as old_specification
from lewm.simulated_body_observation_development import validate_policy_packet
from lewm.physical_execution_development import rotation_xyzw

CANONICAL = {'cluster_00':'family_episode_013', 'cluster_01':'family_episode_083',
    'cluster_02':'family_episode_026', 'cluster_03':'family_episode_003'}
WARMUP_TICKS=3
PREFIX_TICKS=10
BRANCH_TICK=13
SUFFIX_TICKS=40
DRAIN_TICKS=10
COMMAND_TICKS=63


def assignments():
    old=old_assignments();rows=[]
    for cluster,trial in CANONICAL.items():
        role='train' if cluster in ('cluster_00','cluster_01') else 'geometry_transfer'
        cell=old[trial]
        if cell != dict(geometry=cluster+'_left_open',cluster=cluster,data_role=role,
                opening='left_open',appearance_seed=2026090940,action='hold'):
            raise ValueError('exact canonical source geometry assignment required')
        for prefix in ACTIONS:
            for suffix in ACTIONS:
                rows.append(dict(cluster=cluster,geometry_trial=trial,geometry=cell['geometry'],data_role=role,
                    opening=cell['opening'],appearance_seed=cell['appearance_seed'],prefix_action=prefix,suffix_action=suffix))
    random.Random(2026091301).shuffle(rows)
    return {f'switch_episode_{i:03d}':r for i,r in enumerate(rows)}


TRIALS=tuple(assignments())


def specification(trial):
    if trial not in TRIALS:raise ValueError('fixed moving-action cell required')
    return old_specification(assignments()[trial]['geometry_trial'])


def branch_specification(trial):
    if trial not in TRIALS:raise ValueError('fixed moving-action cell required')
    return dict(trial=trial,**assignments()[trial],branch_tick=BRANCH_TICK,
        branch_measured_ns=2_800_000_000,history_observation_indices=[10,11,12,13],
        expected_complete_command_ticks=COMMAND_TICKS,expected_complete_frames=64,
        expected_complete_physics_samples=3900,model_training=False,navigation_qualified=False)


def schedule(trial):
    cell=assignments()[trial]
    rows=[dict(phase=1,role='common_quiet_history',requested_command=[0.,0.,0.]) for _ in range(WARMUP_TICKS)]
    rows += [dict(phase=2,role='moving_prefix',requested_command=list(candidate_commands(cell['prefix_action'])[0])) for _ in range(PREFIX_TICKS)]
    rows += [dict(phase=2 if i<30 else 3,role='candidate_suffix' if i<30 else 'candidate_brake',
        requested_command=list(c)) for i,c in enumerate(candidate_commands(cell['suffix_action']))]
    rows += [dict(phase=3,role='terminal_zero_drain',requested_command=[0.,0.,0.]) for _ in range(DRAIN_TICKS)]
    assert len(rows)==COMMAND_TICKS
    return rows


def decision(trial,tick,policy):
    if type(tick) is not int or not 0<=tick<=COMMAND_TICKS:raise ValueError('bounded branch decision tick required')
    validate_policy_packet(policy);now=1_500_000_000+tick*100_000_000
    if (policy['sensor_state']['decision_ns']!=now or policy['image']['measured_ns']!=now
            or tuple(policy['sensor_state']['identity'])!=(0,0,0)):
        raise ValueError('same actual episode and decision clock required')
    planned=schedule(trial)
    row=planned[tick] if tick<len(planned) else dict(phase=9,role='terminal',requested_command=[0.,0.,0.])
    return row|dict(tick=tick,decision_ns=now,terminal=tick==COMMAND_TICKS,
        tracker_required=False,native_state_used=False,navigation_qualified=False)


def native_horizons(raw,frames):
    start=750+50*BRANCH_TICK-1
    if len(raw['timestamp_s'])<=start:return None
    pose=raw['base_pose_world'];R=rotation_xyzw(pose[start,3:]);origin=pose[start,:3]
    if raw['physics_contact'][:start+1].any():raise ValueError('no post-contact branch labels')
    camera_by_sample={f['physical_sample_index']:i for i,f in enumerate(frames)}
    if start not in camera_by_sample:return None
    assert camera_by_sample[start]==BRANCH_TICK
    targets=[]
    for block in range(1,9):
        at=start+block*250;complete=at<len(pose)
        event=bool(raw['physics_contact'][start+1:min(at+1,len(pose))].any());motion=None
        if complete and not event:
            delta=R.T@(pose[at,:3]-origin);relative=R.T@rotation_xyzw(pose[at,3:])
            motion=[float(delta[0]),float(delta[1]),float(np.arctan2(relative[1,0],relative[0,0]))]
        image=motion is not None and at in camera_by_sample
        targets.append(dict(offset_ns=block*500_000_000,motion_valid=motion is not None,motion=motion,
            contact_valid=complete or event,contact=float(event) if complete or event else None,
            future_image_valid=image,future_observation_index=camera_by_sample[at] if image else None,
            status='CONTACT_EVENT' if event else 'OBSERVED_ENDPOINT' if complete else 'CENSORED_EXECUTION'))
    return dict(departure_tick=BRANCH_TICK,departure_ns=2_800_000_000,
        history_observation_indices=[10,11,12,13],targets=targets,target_only=True,
        label_definition='branch_body_xy_and_projected_relative_yaw;disallowed_native_contact')
