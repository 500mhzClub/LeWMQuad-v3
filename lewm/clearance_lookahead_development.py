"""Follow route bends without aiming a straight shortcut through stored obstacles."""
import numpy as np
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.observed_floor_waypoint_development import centre
from lewm.memory_forecast_clearance_development import TranslationReserveRuntime,select_clear_prediction


def clear_route_target(points,position,cells):
    position=np.asarray(position,float)
    points=[np.asarray(p,float) for p in points]
    geometry=cached_clearance(cells)
    if not points:raise ValueError('nonempty observed route required')
    def clearance(target):
        value=geometry.minimum(position,target)
        return value if value is not None else float('inf')
    start=clearance(position)
    # Preserve available start clearance up to the existing translation reserve.
    # This target-selection rule is not a relaxation of the action-path filter.
    required=min(.48,start)
    target=position.copy();chosen_clearance=start;chosen_index=None
    original=next((p for p in points if np.linalg.norm(p-position)>=.35),points[-1])
    for index,point in enumerate(points):
        value=clearance(point)
        if value+1e-12<required:break
        target=point;chosen_clearance=value;chosen_index=index
        if np.linalg.norm(point-position)>=.35:break
    finite=lambda v:float(v) if np.isfinite(v) else None
    return target,dict(target_map_xy_m=target.tolist(),original_target_map_xy_m=original.tolist(),
        target_changed=bool(np.linalg.norm(target-original)>1e-10),route_index=chosen_index,
        start_clearance_m=finite(start),required_shortcut_clearance_m=required,
        chosen_shortcut_clearance_m=finite(chosen_clearance),
        original_shortcut_clearance_m=finite(clearance(original)),
        no_visible_route_point=chosen_index is None,
        action_clearance_filter_unchanged=True)


class ClearanceLookaheadRuntime(TranslationReserveRuntime):
    def _route_target(self,route,snapshot,position):
        points=[centre(c) for c in route['route_cells']]
        if route.get('status')=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL' and 'goal_map_xy_m' in route:
            points.append(np.asarray(route['goal_map_xy_m'],float))
        target,receipt=clear_route_target(points,
            position,snapshot.fine_occupied)
        route['lookahead']=receipt
        return target


class ReserveRecoveryLookaheadRuntime(ClearanceLookaheadRuntime):
    def _select_clear_prediction(self,selected,prediction,snapshot,position,rotation):
        return select_clear_prediction(selected,prediction,snapshot.fine_occupied,position,rotation,
            translation_reserve_m=.03,reserve_recovery=True)


class StandOffFrontierRuntime(ReserveRecoveryLookaheadRuntime):
    frontier_standoff_m=.50
