"""Separate causal suffix derivation; never relabel the frozen initial targets.

Caller authenticates complete family raw/audit artifacts before deriving labels.
Native state is consumed by derive only. materialize accepts recorded policy
packets plus segregated labels, with no native state in model inputs.
"""
import numpy as np
import torch
from lewm.geometry_progress_layout_family_development import assignments,candidate_commands
from lewm.physical_execution_development import rotation_xyzw
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.rgb_body_tensor_interface_development import observation_tensors

OFFSETS_TICKS=tuple(range(0,40,5))


def remaining_candidate(action,offset_ticks):
    if type(offset_ticks) is not int or offset_ticks not in OFFSETS_TICKS:raise ValueError('fixed causal suffix offset required')
    command=torch.tensor(candidate_commands(action)[offset_ticks:],dtype=torch.float32)/torch.tensor([.3,1.,.5])
    values=torch.zeros((40,3));values[:len(command)]=command
    valid=torch.arange(40)<len(command)
    return values.reshape(8,5,3),valid.reshape(8,5)


def assignment(row):
    if row['trial'] not in assignments() or any(row.get(k)!=v for k,v in assignments()[row['trial']].items()):
        raise ValueError('exact family role/cluster/action assignment required')
    return assignments()[row['trial']]


def derive(raw,cameras,report):
    cell=assignment(report)
    if not report['raw_sensor_reconstruction_pass'] or not report['command_stop_replay_pass']:
        raise ValueError('completed raw family replay required')
    n=len(raw['timestamp_s']);pose=raw['base_pose_world'];contact=raw['physics_contact']
    if (not 0<n<=2900 or pose.shape!=(n,7) or contact.shape!=(n,)
            or not np.isfinite(pose).all() or not np.array_equal(np.rint(raw['timestamp_s']*1e9).astype(np.int64),
                np.arange(1,n+1)*2_000_000)):
        raise ValueError('complete finite bounded native trace and exact clocks required')
    samples=[c['physical_sample_index'] for c in cameras]
    if samples!=[749+50*i for i in range(len(cameras))] or any(i>=n for i in samples):
        raise ValueError('complete ordered actual camera prefix required')
    camera_by_sample={sample:i for i,sample in enumerate(samples)};rows=[]
    for offset in OFFSETS_TICKS:
        frame=3+offset;start=749+50*frame;now=1_500_000_000+frame*100_000_000
        row=dict(window_id=f"{report['trial']}/offset_{offset:02d}",trial=report['trial'],**cell,
            offset_ticks=offset,remaining_ticks=40-offset,decision_ns=now,
            history_observation_indices=list(range(frame-3,frame+1)),target_only=True,
            available=False,targets=None,reason=None)
        if start>=n or frame>=len(cameras):row['reason']='MISSING_ACTUAL_CONTEXT';rows.append(row);continue
        if contact[:start+1].any():row['reason']='POST_CONTACT_CONTEXT';rows.append(row);continue
        R=rotation_xyzw(pose[start,3:]);origin=pose[start,:3];targets=[]
        for block in range(1,9):
            in_plan=block*5<=40-offset;at=start+block*250
            complete=in_plan and at<n
            event=bool(in_plan and contact[start+1:min(at+1,n)].any());motion=None
            if complete and not event:
                delta=R.T@(pose[at,:3]-origin);relative=R.T@rotation_xyzw(pose[at,3:])
                motion=[float(delta[0]),float(delta[1]),float(np.arctan2(relative[1,0],relative[0,0]))]
            image=in_plan and at in camera_by_sample
            targets.append(dict(offset_ns=block*500_000_000 if in_plan else 0,in_plan=in_plan,
                motion_valid=motion is not None,motion=motion,
                contact_valid=complete or event,contact=float(event) if complete or event else None,
                future_image_valid=image,future_observation_index=camera_by_sample[at] if image else None))
        row.update(available=True,targets=targets);rows.append(row)
    return rows


def materialize(reader,row):
    assignment(row);offset=row['offset_ticks'];blocks,valid=remaining_candidate(row['action'],offset)
    frame=3+offset;now=1_500_000_000+frame*100_000_000
    if (row['available'] is not True or row['target_only'] is not True or row['reason'] is not None
            or row['decision_ns']!=now or row['remaining_ticks']!=40-offset
            or row['history_observation_indices']!=list(range(frame-3,frame+1)) or len(row['targets'])!=8):
        raise ValueError('exact available causal context and target population required')
    packets=[reader.packet(i)[0] for i in row['history_observation_indices']]
    history=causal_history_tensors(packets,now)
    future={k:torch.zeros((8,*v.shape[1:]),dtype=v.dtype) for k,v in history.items()}
    motion=torch.full((8,3),float('nan'));contact=torch.full((8,),float('nan'))
    mv=torch.zeros(8,dtype=torch.bool);cv=mv.clone();fv=mv.clone();offsets=torch.zeros(8,dtype=torch.int64)
    positive_seen=False
    for i,t in enumerate(row['targets']):
        in_plan=(i+1)*5<=40-offset;ns=(i+1)*500_000_000 if in_plan else 0
        if (type(t['in_plan']) is not bool or t['in_plan']!=in_plan or t['offset_ns']!=ns
                or any(type(t[k]) is not bool for k in ('motion_valid','contact_valid','future_image_valid'))
                or not in_plan and any(t[k] for k in ('motion_valid','contact_valid','future_image_valid'))):
            raise ValueError('exact remaining-plan clocks and independent boolean masks required')
        offsets[i]=ns;mv[i]=t['motion_valid'];cv[i]=t['contact_valid'];fv[i]=t['future_image_valid']
        if t['contact_valid']:
            if type(t['contact']) is not float or t['contact'] not in (0.,1.) or positive_seen and t['contact']==0.:
                raise ValueError('measured cumulative contact cannot revert')
            positive_seen|=t['contact']==1.;contact[i]=t['contact']
        elif t['contact'] is not None:raise ValueError('unknown contact must remain null')
        if t['motion_valid']:
            values=np.asarray(t['motion'],float)
            if not t['contact_valid'] or t['contact']!=0. or values.shape!=(3,) or not np.isfinite(values).all():
                raise ValueError('finite contact-free native motion required')
            motion[i]=torch.tensor(values,dtype=torch.float32)
        elif t['motion'] is not None:raise ValueError('missing motion must remain null')
        index=t['future_observation_index']
        if t['future_image_valid']:
            if type(index) is not int or index!=frame+5*(i+1):raise ValueError('actual same-episode boundary image required')
            packet=reader.packet(index)[0]
            if (packet['image']['measured_ns']!=now+ns or packet['sensor_state']['decision_ns']!=now+ns
                    or packet['sensor_state']['identity']!=packets[-1]['sensor_state']['identity']):
                raise ValueError('actual same-episode future clock required')
            for k,v in observation_tensors(packet).items():future[k][i]=v
        elif index is not None:raise ValueError('missing future cannot substitute an index')
    return dict(inputs=dict(observation_history=history,known_action_blocks=blocks,known_action_valid=valid),
        targets=dict(motion=motion,motion_valid=mv,contact=contact,contact_valid=cv,
            future_observations=future,future_valid=fv,target_offsets_ns=offsets))
