"""Audited native traces -> target-only labels, never policy/model inputs.

Contact means the trace's disallowed-native-contact event, not all foot forces
or calibrated terrain risk. A noncontact stop never proves future safety.
RGB availability and visual tracking are deliberately not motion-label gates.
"""
import numpy as np
import torch
from lewm.physical_execution_development import rotation_xyzw
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan,validate_timed_plan

DT_NS=2_000_000


class PulseNativeTargets:
    def __init__(self,raw):
        self.ns=np.rint(np.asarray(raw['timestamp_s'])*1e9).astype(np.int64)
        self.pose=np.asarray(raw['base_pose_world'],float)
        commands=np.asarray(raw['requested_command'])
        self.contact=np.asarray(raw['physics_contact'])
        n=len(self.ns)
        if (self.ns.ndim!=1 or n<750 or not np.array_equal(self.ns,np.arange(1,n+1)*DT_NS)
                or self.pose.shape!=(n,7) or not np.isfinite(self.pose).all()
                or commands.shape!=(n,3) or not np.isfinite(commands).all()
                or self.contact.shape!=(n,) or self.contact.dtype!=np.bool_
                or not np.allclose(np.linalg.norm(self.pose[:,3:],axis=1),1.,rtol=0,atol=1e-5)):
            raise ValueError('complete finite native trace with unit xyzw and boolean contacts required')
        # Fixed requests are stored at native float32 precision. This compares
        # command identity, not tolerance on physical outcomes or velocity.
        self.commands=commands.astype(np.float32)

    def labels(self,window):
        now=window['decision_ns'];start=now//DT_NS-1
        if (type(now) is not int or now%DT_NS or not 749<=start<len(self.ns)
                or now!=1_500_000_000+window['departure_tick']*100_000_000):
            raise ValueError('actual departure sample required')
        blocks,mask=pulse_brake_plan(tuple(window['command']),window['pulse_ticks'])
        active,offsets=validate_timed_plan(blocks[None],mask[None],1)
        if len(window['targets'])!=8:raise ValueError('all target slots required')
        expected=np.asarray([window['command']]*window['pulse_ticks']+[[0.,0.,0.]]*20,np.float32)
        expected=np.repeat(expected,50,axis=0)
        actual=self.commands[start+1:start+1+len(expected)]
        matches=np.all(actual==expected[:len(actual)],axis=1)
        mismatch=np.flatnonzero(~matches)
        count=int(mismatch[0]) if len(mismatch) else len(actual)
        matched_end=start+count
        matched_until=int(self.ns[matched_end])
        past_contact=bool(self.contact[:start+1].any())
        events=np.flatnonzero(self.contact[start+1:matched_end+1])
        first_event=int(self.ns[start+1+events[0]]) if len(events) else None
        R0=rotation_xyzw(self.pose[start,3:]);origin=self.pose[start,:3]
        motion=torch.full((8,3),float('nan'));contact=torch.full((8,),float('nan'))
        mv=torch.zeros(8,dtype=torch.bool);cv=mv.clone();reasons=[]
        for i,t in enumerate(window['targets']):
            offset=int(offsets[0,i]);target=now+offset if active[0,i] else None
            if t['offset_ns']!=offset or t['target_ns']!=target:raise ValueError('native target timing mismatch')
            if not active[0,i]:reasons.append('UNKNOWN_PLAN');continue
            if past_contact:reasons.append('CONTACT_BEFORE_DEPARTURE');continue
            # Positive contact can be known before an interrupted endpoint;
            # a contact occurring after command divergence is not plan evidence.
            event=first_event is not None and first_event<=target
            complete=bool(t['command_prefix_executed']) and target<=matched_until
            cv[i]=complete or event
            if cv[i]:contact[i]=float(event)
            mv[i]=complete and not event
            if mv[i]:
                at=target//DT_NS-1;delta=R0.T@(self.pose[at,:3]-origin)
                relative=R0.T@rotation_xyzw(self.pose[at,3:])
                motion[i]=torch.tensor([delta[0],delta[1],np.arctan2(relative[1,0],relative[0,0])])
            reasons.append('CONTACT_EVENT' if event else 'OBSERVED_ENDPOINT' if complete else 'CENSORED_EXECUTION')
        return dict(motion=motion,motion_valid=mv,contact=contact,contact_valid=cv,
                    target_offsets_ns=offsets[0],accounting=dict(matched_until_ns=matched_until,
                        first_matched_contact_ns=first_event,contact_before_departure=past_contact,target_status=reasons),
                    label_definition='current_body_xy_and_projected_relative_yaw;disallowed_native_contact',
                    target_only=True)
