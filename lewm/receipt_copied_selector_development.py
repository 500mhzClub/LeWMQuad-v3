"""Separate selector bindings using identical code with a faster receipt copier.

No imported module, original function namespace or live object is patched.
Each fork retains the original code, closure and defaults. Only the explicitly
listed global dependencies differ. The mirrored MRO routes original super()
calls through these forks while keeping the original initialization/state.
"""
from types import FunctionType
from copy import deepcopy
from lewm.receipt_copy_development import copy_receipt
from lewm.observation_horizon_waypoint_selection_development import scan_rank, restrict
from lewm.observation_horizon_surface_filter_development import filter_selection
from lewm.observation_horizon_waypoint_utility_development import score_commitment_pose
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.eight_step_planning_development import plan
from lewm.executed_horizon_final_goal_development import score_final_goal
from lewm.causal_residual_final_goal_development import correct_final_goal_score
from lewm.nominal_clearance_reentry_development import reenter
from lewm.view_reentry_selection_development import reenter_with_translation
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.mission_target_waypoint_selection_development import MissionTargetWaypointSelector
from lewm.mission_target_eight_step_selection_development import MissionTargetEightStepSelector
from lewm.observed_round_trip_controller_development import RoundTripMissionSelector
from lewm.view_reentry_round_trip_controller_development import ViewReentrySelector
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController

FORKS=[]


def fork(original,*,replace_copy=True,**dependencies):
    replacements=dict(dependencies)
    if replace_copy:
        if original.__globals__.get('deepcopy') is not deepcopy:
            raise ValueError('original standard copier binding required')
        replacements['deepcopy']=copy_receipt
    if any(name not in original.__globals__ for name in replacements):
        raise ValueError('only existing explicit function dependencies may change')
    namespace=original.__globals__.copy(); namespace.update(replacements)
    result=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    result.__kwdefaults__=original.__kwdefaults__
    result.__annotations__=original.__annotations__
    result.__qualname__=original.__qualname__
    result.__doc__=original.__doc__
    FORKS.append((original,result,replacements))
    return result


copied_scan_rank=fork(scan_rank)
copied_restrict=fork(restrict)
copied_filter_selection=fork(filter_selection)
copied_score_commitment_pose=fork(score_commitment_pose)
copied_constrain=fork(constrain)
copied_plan=fork(plan)
copied_score_final_goal=fork(score_final_goal)
copied_correct_final_goal_score=fork(correct_final_goal_score)
copied_reenter=fork(reenter)
copied_reenter_with_translation=fork(reenter_with_translation,reenter=copied_reenter)
copied_score_waypoint_execution=fork(score_waypoint_execution)


class ReceiptCopiedMissionTargetSelector(MissionTargetWaypointSelector):
    choose=fork(MissionTargetWaypointSelector.choose,replace_copy=False,
        scan_rank=copied_scan_rank,restrict=copied_restrict,filter_selection=copied_filter_selection)


class ReceiptCopiedEightStepSelector(MissionTargetEightStepSelector,ReceiptCopiedMissionTargetSelector):
    choose=fork(MissionTargetEightStepSelector.choose,replace_copy=False,
        restrict=copied_restrict,score_commitment_pose=copied_score_commitment_pose,
        constrain=copied_constrain,plan=copied_plan)


class ReceiptCopiedRoundTripSelector(RoundTripMissionSelector,ReceiptCopiedEightStepSelector):
    choose=fork(RoundTripMissionSelector.choose,replace_copy=False,
        score_final_goal=copied_score_final_goal,correct_final_goal_score=copied_correct_final_goal_score)


class ReceiptCopiedViewReentrySelector(ViewReentrySelector,ReceiptCopiedRoundTripSelector):
    choose=fork(ViewReentrySelector.choose,replace_copy=False,
        reenter_with_translation=copied_reenter_with_translation,score_waypoint_execution=copied_score_waypoint_execution)


class ReceiptCopiedMeasuredFloorController(MeasuredFloorTransportController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.selector=ReceiptCopiedViewReentrySelector(residual=self.residual,
            condition=self.selector.condition,variant=self.selector.variant,
            goal_initial_body_xy_m=self.mission.target())
