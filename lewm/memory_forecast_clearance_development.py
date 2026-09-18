"""Filter learned candidate paths using previously observed fine obstacle cells."""
import numpy as np
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.frontier_visit_runtime_development import FrontierVisitRuntime
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands


def reserve_recovery_clear(distances,required):
    """Leave an existing reserve deficit without predicting further encroachment."""
    if len(distances)!=8:raise ValueError('three-prefix/four-commit/one-stop horizon required')
    if any(d is None for d in distances):return False
    prefix=min(distances[:3])
    return bool(.45+1e-12<prefix<=required+1e-12
        and min(distances[3:])+1e-12>=prefix
        and distances[7]>required+1e-12)


def select_clear_prediction(selected,prediction,cells,position,rotation,*,translation_reserve_m=0.,
        reserve_recovery=False):
    if not np.isfinite(translation_reserve_m) or not 0<=translation_reserve_m<=.10:
        raise ValueError('explicit bounded development translation reserve required')
    clearance=cached_clearance(cells)
    candidates=[]
    for i,action in enumerate(ACTIONS):
        points=np.asarray(position)[:2]+np.vstack((np.zeros(2),prediction[i,:,:2]))@np.asarray(rotation)[:2,:2].T
        minimum=None;distances=[]
        for begin,end in zip(points[:-1],points[1:]):
            value=clearance.minimum(begin,end)
            distances.append(value)
            if value is not None:
                minimum=value if minimum is None else min(minimum,value)
        reserve=translation_reserve_m if any(candidate_commands(action)[0][:2]) else 0.
        required=.45+reserve
        full_clear=minimum is None or minimum>required+1e-12
        recovery=bool(reserve_recovery and reserve>0 and not full_clear
            and reserve_recovery_clear(distances,required))
        receipt=dict(action=action,minimum_predicted_path_clearance_m=minimum,
            prediction_error_reserve_m=reserve,required_path_clearance_m=required,
            nominal_predicted_path_clear=full_clear or recovery)
        if reserve_recovery:
            receipt.update(full_reserve_path_clear=full_clear,reserve_recovery_path_clear=recovery,
                nominal_footprint_path_clear=minimum is None or minimum>.45+1e-12,
                segment_clearances_m=distances,
                clearance_check_mode='FULL_RESERVE' if full_clear else 'RESERVE_RECOVERY' if recovery else 'BLOCKED')
        candidates.append(receipt)
    utilities={r['action']:r['utility_m'] for r in selected.get('scan_utilities',selected['candidates'])}
    eligible=[i for i,r in enumerate(candidates) if r['action'] in utilities and r['nominal_predicted_path_clear']]
    index=max(eligible,key=lambda i:utilities[ACTIONS[i]]) if eligible else ACTIONS.index('hold')
    action=ACTIONS[index]
    result=selected|dict(action=action,action_index=index,requested_command=candidate_commands(action)[0],
        before_memory_filter_action=selected['action'],memory_forecast_candidates=candidates,
        memory_forecast_status='CLEAR_CANDIDATE_SELECTED' if eligible else 'NO_CLEAR_CANDIDATE_ZERO_REQUESTED',
        memory_footprint_radius_m=.45,memory_forecast_endpoint_ns=800_000_000,
        translation_prediction_error_reserve_m=translation_reserve_m,
        predicted_clearance_is_not_execution_certificate=True)
    if reserve_recovery:
        result.update(reserve_recovery_enabled=True,
            selected_reserve_recovery=bool(candidates[index]['reserve_recovery_path_clear']),
            recovery_preserves_nominal_footprint=True,recovery_restores_reserve_by_commit_end=True)
    return result


class MemoryForecastClearanceRuntime(FrontierVisitRuntime):
    def _select_clear_prediction(self,selected,prediction,snapshot,position,rotation):
        return select_clear_prediction(selected,prediction,snapshot.fine_occupied,position,rotation)


class TranslationReserveRuntime(MemoryForecastClearanceRuntime):
    def _select_clear_prediction(self,selected,prediction,snapshot,position,rotation):
        return select_clear_prediction(selected,prediction,snapshot.fine_occupied,position,rotation,
            translation_reserve_m=.03)
