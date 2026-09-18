"""Causal comparison until the first changed request or either terminal state."""
from copy import deepcopy
import numpy as np

MAX_FRAMES=646


def without_retention(value):
    if isinstance(value,dict):
        return {k:without_retention(v) for k,v in value.items() if k!='recent_qualified_anchor'}
    if isinstance(value,list):return [without_retention(v) for v in value]
    return value


class PrefixComparison:
    def __init__(self):
        self.next_frame=0;self.stopped=False;self.last_qualified_frame=None
        self.first_reference_attempt=None;self.first_qualified_reference=None
        self.first_decision_difference=None;self.first_command_difference=None

    def compare(self,original,candidate,actual_command,*,frame):
        if (self.stopped or type(frame) is not int or frame!=self.next_frame or not 0<=frame<MAX_FRAMES
                or original['controller']!='partial_height_direct_flow_controller_v1'
                or candidate['controller']!='recent_qualified_anchor_controller_v1'
                or candidate['recent_qualified_anchor_enabled'] is not True
                or candidate['partial_floor_height_constraint_enabled'] is not True
                or original['requested_command']!=actual_command):
            raise ValueError('ordered original height episode and explicit retention successor required')
        now=1_500_000_000+frame*100_000_000
        visual=candidate['original_visual_evidence'];receipt=visual['recent_qualified_anchor']
        if (receipt['retained_from_frame']!=self.last_qualified_frame
                or receipt['maximum_additional_references']!=1
                or receipt['original_reference_population_preserved_before_search'] is not True
                or receipt['reference_retained_from_bridge'] is not False
                or receipt['reference_retained_from_floor_transport'] is not False
                or receipt['bridge_limit_unchanged'] is not True
                or receipt['pose_uncertainty_calibrated'] is not False):
            raise ValueError('bounded causal retention of a previously qualified visual view required')
        attempts=receipt['attempts']
        if type(attempts) is not list or len(attempts)>4:
            raise ValueError('at most two cameras in each original/direct-flow pass required')
        for a in attempts:
            selection=a['original_reference_selection']
            if (a['camera'] not in ('primary','auxiliary') or a['current_frame']!=frame
                    or a['reference_frame']!=frame-1 or a['reference_frame']!=self.last_qualified_frame
                    or a['reference_measured_ns']!=now-100_000_000
                    or a['reference_was_anchor_qualified'] is not True
                    or a['reference_is_immediately_previous'] is not True
                    or a['rigid_thresholds_unchanged'] is not True
                    or type(a['qualified']) is not bool or type(a['direct_flow_mode']) is not bool
                    or selection['status']!='NO_QUALIFIED_REFERENCE'
                    or not 1<=selection['retained_references']<=8
                    or len(selection['attempts'])!=selection['retained_references']
                    or any(r['status']!='REJECTED' for r in selection['attempts'])
                    or any(r['reference_frame']==a['reference_frame'] for r in selection['attempts'])
                    or (a['qualified'] and a['failure'] is not None)
                    or (not a['qualified'] and not isinstance(a['failure'],str))):
                raise ValueError('each extra pair requires original missingness and a causal qualified reference')
        if attempts and self.first_reference_attempt is None:self.first_reference_attempt=frame
        qualified=sum(a['qualified'] for a in attempts)
        if qualified and self.first_qualified_reference is None:self.first_qualified_reference=frame
        continuity=visual.get('continuity_evidence') or {}
        pose=visual.get('current_pose')
        retained=receipt['retained_current_frame']
        if retained is not None:
            if (retained!=frame or visual['status']!='CURRENT_VISUAL_POSE' or pose is None
                    or pose['frame']!=frame or pose['measured_ns']!=now
                    or continuity['status'] not in ('INITIAL_REFERENCE','ANCHOR_MEASUREMENT')):
                raise ValueError('a current accepted anchor measurement must precede retention')
        self.last_qualified_frame=retained
        normalized=without_retention(candidate)
        normalized.pop('recent_qualified_anchor_enabled')
        normalized['controller']=original['controller']
        exact=normalized==original
        if not exact and self.first_reference_attempt is None:
            raise ValueError('complete original decision must match before any extra reference attempt')
        if not exact and self.first_decision_difference is None:self.first_decision_difference=frame
        old_selection=original.get('new_selection') or {};new_selection=candidate.get('new_selection') or {}
        compared='prediction' in old_selection and 'prediction' in new_selection
        if compared:
            old=np.asarray(old_selection['prediction']);new=np.asarray(new_selection['prediction'])
            if old.shape!=(6,8,5) or not np.isfinite(old).all() or not np.array_equal(old,new):
                raise ValueError('identical raw forecast banks on shared observations required')
        request=np.asarray(candidate['requested_command'],float)
        if request.shape!=(3,) or not np.isfinite(request).all():raise ValueError('finite three-axis command required')
        changed=candidate['requested_command']!=actual_command
        if changed and self.first_command_difference is None:self.first_command_difference=frame
        stop=changed or original['terminal'] is not None or candidate['terminal'] is not None
        if candidate['terminal'] is not None and candidate['requested_command']!=[0.,0.,0.]:
            raise ValueError('terminal state must request zero command')
        self.next_frame+=1;self.stopped=stop
        return dict(frame=frame,complete_original_decision_exact=exact,
            extra_reference_attempts=len(attempts),extra_qualified_references=qualified,
            raw_model_forecasts_compared=compared,requested_command_changed=changed,
            terminal_changed=original['terminal']!=candidate['terminal'],stop=stop,
            first_reference_attempt=self.first_reference_attempt,
            first_qualified_reference=self.first_qualified_reference,
            first_decision_difference=self.first_decision_difference,
            first_command_difference=self.first_command_difference,
            following_recorded_observations_consumed=False,unexecuted_outcomes_inferred=False)
