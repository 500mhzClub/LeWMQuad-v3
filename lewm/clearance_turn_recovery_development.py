"""Reserve space for turn drift and finish a clear alternative turning direction."""
from copy import deepcopy
import math
from lewm.initial_panorama_development import InitialSurveyRuntime
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.memory_forecast_clearance_development import reserve_recovery_clear

TURN_RESERVE_M=.03


def wrap(value):return math.atan2(math.sin(value),math.cos(value))


def choose(result,index):
    result.update(action=ACTIONS[index],action_index=index,
        requested_command=candidate_commands(ACTIONS[index])[0],
        selected_reserve_recovery=bool(result['memory_forecast_candidates'][index]['reserve_recovery_path_clear']),
        selected_stepwise_recovery=bool(result['memory_forecast_candidates'][index].get('stepwise_reserve_recovery_path_clear',False)))
    return result


def stepwise_recovery_clear(distances,required):
    if len(distances)!=8 or any(d is None for d in distances):return False
    prefix=min(distances[:3]);gain=max(.001,.1*(required-prefix))
    return bool(.45+1e-12<prefix<=required+1e-12
        and min(distances[3:])+1e-12>=prefix and distances[-1]>=prefix+gain)


def reserve_turns(selection,*,stepwise=False):
    result=deepcopy(selection)
    for row in result['memory_forecast_candidates']:
        if row['action'] not in ('left_turn','right_turn'):continue
        required=.45+TURN_RESERVE_M
        full=row['minimum_predicted_path_clearance_m'] is None or row['minimum_predicted_path_clearance_m']>required+1e-12
        recovery=not full and reserve_recovery_clear(row['segment_clearances_m'],required)
        gradual=bool(stepwise and not full and stepwise_recovery_clear(row['segment_clearances_m'],required))
        row.update(prediction_error_reserve_m=TURN_RESERVE_M,required_path_clearance_m=required,
            full_reserve_path_clear=full,reserve_recovery_path_clear=bool(recovery or gradual),
            stepwise_reserve_recovery_path_clear=gradual,
            nominal_predicted_path_clear=bool(full or recovery or gradual),
            clearance_check_mode='FULL_RESERVE' if full else 'RESERVE_RECOVERY' if recovery else
                'STEPWISE_RESERVE_RECOVERY' if gradual else 'BLOCKED')
    utilities={r['action']:r['utility_m'] for r in result.get('scan_utilities',result['candidates'])}
    eligible=[i for i,r in enumerate(result['memory_forecast_candidates'])
        if r['action'] in utilities and r['nominal_predicted_path_clear']]
    index=max(eligible,key=lambda i:utilities[ACTIONS[i]]) if eligible else ACTIONS.index('hold')
    result.update(turn_prediction_error_reserve_m=TURN_RESERVE_M,
        memory_forecast_status='CLEAR_CANDIDATE_SELECTED' if eligible else 'NO_CLEAR_CANDIDATE_ZERO_REQUESTED')
    if stepwise:
        result.update(stepwise_reserve_recovery_enabled=True,recovery_restores_reserve_by_commit_end=False,
            recovery_minimum_gain_m=.001,recovery_minimum_fraction_of_remaining_deficit=.1,
            stepwise_recovery_nominal_footprint_unchanged=True)
    return choose(result,index)


def recover_turn(selection,heading,generation,state,*,stepwise=False,release_for_progress=False):
    result=reserve_turns(selection,stepwise=stepwise)
    if state is not None and state['mission_generation']!=generation:state=None
    if release_for_progress and state is not None and 'scan_utilities' not in result:
        action=result['action']
        candidates={r['action']:r for r in result['candidates']}
        clearance=next(r for r in result['memory_forecast_candidates'] if r['action']==action)
        if (action in ('forward','left_arc','right_arc') and clearance['full_reserve_path_clear']
                and candidates[action]['predicted_progress_during_commit_m']>0.
                and candidates[action]['position_contact_utility_m']>candidates['hold']['position_contact_utility_m']):
            result['clearance_turn']=dict(active=False,event='FULL_RESERVE_PROGRESS_REJOINS_ROUTE',
                previous_target_heading_rad=state['target_heading_rad'],
                released_action=action,prediction_and_clearance_checks_unchanged=True)
            return result,None
    event=None
    if state is not None:
        error=wrap(state['target_heading_rad']-heading)
        remaining=(state['direction']*error)%(2*math.pi)
        crossed=remaining-state['previous_remaining_rad']>math.pi
        if abs(error)<=.1 or crossed:
            state=None;event='MEASURED_TARGET_HEADING_REACHED'
        else:state=state|dict(previous_remaining_rad=remaining)
    preferred=result['before_memory_filter_action']
    by_action={r['action']:r for r in result['memory_forecast_candidates']}
    # A blocked translating arc can require the same long-way turn as a
    # blocked pure turn. Otherwise the only improving-clearance turn loses
    # to hold because its immediate waypoint-alignment utility is negative.
    if stepwise and result['action']=='hold' and preferred in ('forward','left_arc','right_arc'):
        goal=result['waypoint_body_xy_m']
        error=math.atan2(goal[1],goal[0])
        if abs(error)>.1:
            preferred='left_turn' if error>0 else 'right_turn'
    if state is None and event is None and preferred in ('left_turn','right_turn'):
        alternative='right_turn' if preferred=='left_turn' else 'left_turn'
        error=result.get('scan_heading_error_rad')
        if error is None:
            goal=result['waypoint_body_xy_m']
            error=math.atan2(goal[1],goal[0])
        if (abs(error)>.1 and not by_action[preferred]['nominal_predicted_path_clear']
                and by_action[alternative]['nominal_predicted_path_clear']):
            direction=1 if alternative=='left_turn' else -1
            state=dict(target_heading_rad=wrap(heading+error),direction=direction,
                previous_remaining_rad=(direction*error)%(2*math.pi),
                mission_generation=generation,blocked_preferred_action=preferred)
            event='CLEAR_ALTERNATIVE_TURN_LATCHED'
    if state is not None:
        action='left_turn' if state['direction']==1 else 'right_turn'
        clear=by_action[action]['nominal_predicted_path_clear']
        other='right_turn' if action=='left_turn' else 'left_turn'
        full_alternative=by_action[other].get('full_reserve_path_clear',False)
        if stepwise and not clear and (full_alternative or by_action[other].get('stepwise_reserve_recovery_path_clear')):
            direction=-state['direction']
            state=state|dict(direction=direction,
                previous_remaining_rad=(direction*wrap(state['target_heading_rad']-heading))%(2*math.pi),
                reserve_recovery_direction_switches=state.get('reserve_recovery_direction_switches',0)+1)
            action=other;clear=True
            event='FULL_RESERVE_ALTERNATIVE_DIRECTION_SELECTED' if full_alternative else 'CLEARANCE_INCREASING_RECOVERY_DIRECTION_SELECTED'
        choose(result,ACTIONS.index(action if clear else 'hold'))
        result['clearance_turn']=state|dict(active=True,event=event,
            latched_turn_forecast_clear=clear,completion_uses_measured_heading=True)
    elif event is not None:result['clearance_turn']=dict(active=False,event=event)
    return result,state


class ClearanceTurnRecoveryRuntime(InitialSurveyRuntime):
    stepwise_recovery=False
    release_for_progress=False
    def __init__(self,*args,**kwargs):
        self.clearance_turn=None
        super().__init__(*args,**kwargs)

    def _select_clear_prediction(self,selected,prediction,snapshot,position,rotation):
        original=super()._select_clear_prediction(selected,prediction,snapshot,position,rotation)
        heading=math.atan2(rotation[1,0],rotation[0,0])
        result,self.clearance_turn=recover_turn(original,heading,self.mission_generation,self.clearance_turn,
            stepwise=self.stepwise_recovery,release_for_progress=self.release_for_progress)
        return result


class StepwiseClearanceTurnRecoveryRuntime(ClearanceTurnRecoveryRuntime):
    stepwise_recovery=True
