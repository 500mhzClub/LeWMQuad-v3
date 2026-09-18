"""Target-side native derivation at actual 100-ms observation boundaries.

The caller authenticates source assignments, raw audits and complete population
accounting. This helper neither reads files nor supplies policy/model inputs.
"""
import numpy as np
from lewm.physical_execution_development import rotation_xyzw


def derive(raw,cameras,*,frame,commands):
    n=len(raw['timestamp_s']);pose=raw['base_pose_world'];contact=raw['physics_contact']
    commands=np.asarray(commands,float)
    if (type(frame) is not int or frame<3 or commands.ndim!=2 or commands.shape[1]!=3
            or not 1<=len(commands)<=8 or not np.isfinite(commands).all()
            or np.any(commands[:,1]) or np.any(np.abs(commands)>[.3,0.,.5])):
        raise ValueError('actual causal context and bounded known command prefix required')
    if (not 0<n<=3900 or pose.shape!=(n,7) or contact.shape!=(n,)
            or raw['requested_command'].shape!=(n,3) or not np.isfinite(pose).all()
            or not np.isin(contact,[0,1]).all()
            or not np.array_equal(np.rint(raw['timestamp_s']*1e9).astype(np.int64),
                np.arange(1,n+1)*2_000_000)):
        raise ValueError('finite complete native trace and exact 2-ms clocks required')
    samples=[c['physical_sample_index'] for c in cameras]
    if samples!=[749+50*i for i in range(len(cameras))] or any(i>=n for i in samples):
        raise ValueError('complete ordered actual camera prefix required')
    start=749+50*frame;now=1_500_000_000+frame*100_000_000
    common=dict(target_only=True,departure_tick=frame,departure_ns=now,
        history_observation_indices=list(range(frame-3,frame+1)),
        target_cadence_ns=100_000_000,maximum_horizon_ns=800_000_000)
    if start>=n or frame>=len(cameras):
        return common|dict(available=False,reason='MISSING_ACTUAL_CONTEXT',targets=None)
    if contact[:start+1].any():
        return common|dict(available=False,reason='POST_CONTACT_CONTEXT',targets=None)
    R=rotation_xyzw(pose[start,3:]);origin=pose[start,:3];targets=[]
    for h in range(1,9):
        active=h<=len(commands);at=start+h*50;complete=active and at<n
        if active:
            a=start+(h-1)*50+1;b=min(at+1,n)
            if a<b:
                np.testing.assert_array_equal(raw['requested_command'][a:b],np.tile(commands[h-1],(b-a,1)))
        event=bool(active and contact[start+1:min(at+1,n)].any());motion=None
        if complete and not event:
            delta=R.T@(pose[at,:3]-origin);relative=R.T@rotation_xyzw(pose[at,3:])
            if np.hypot(relative[1,0],relative[0,0])<=1e-8:raise ValueError('undefined actual relative yaw')
            motion=[float(delta[0]),float(delta[1]),float(np.arctan2(relative[1,0],relative[0,0]))]
        future=bool(active and motion is not None and frame+h<len(cameras))
        targets.append(dict(offset_ns=h*100_000_000 if active else 0,in_plan=active,
            motion_valid=motion is not None,motion=motion,
            contact_valid=complete or event,contact=float(event) if complete or event else None,
            future_image_valid=future,future_observation_index=frame+h if future else None))
    return common|dict(available=True,reason=None,targets=targets)


def verify_half_second_overlap(short,original):
    if short['available'] is not True or short['targets'] is None or original is None:
        raise ValueError('two actual available target populations required')
    new,old=short['targets'][4],original[0]
    for key in ('offset_ns','in_plan','motion_valid','motion','contact_valid','contact'):
        if new[key]!=old[key]:raise ValueError('changed shared 500-ms native target: '+key)
    if new['future_image_valid']:
        if not old['future_image_valid'] or new['future_observation_index']!=old['future_observation_index']:
            raise ValueError('changed shared actual 500-ms future observation')
    return True
