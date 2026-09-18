"""Compare fine-goal decisions and timing on saved reconstructed sensor maps."""
from copy import deepcopy
import json
import time
from types import SimpleNamespace
import numpy as np

from lewm.cached_fine_goal_route_development import cached_fine_goal_route,cached_graph_geometry
from lewm.fine_stored_obstacle_routing_development import _cached_clearance
from lewm.cached_fine_connectivity_development import fine_goal_route,search_graph,floor_index
from lewm.clearance_preferred_route_development import refine_proposal
from scripts.run_go2_current_position_coverage_view_development import BASE,ROOT


def without_time(value):
    value=deepcopy(value)
    if 'fine_goal_route' in value:value['fine_goal_route'].pop('routing_s')
    return value


def main():
    root=BASE/ROOT;output=root/'cached_fine_connectivity_comparison_v1.json'
    if output.exists():raise ValueError('preserve completed comparison')
    saved=json.loads((root/'routing_profile_v1/result.json').read_text())['rows']
    rows=[]
    for row in saved:
        snapshot=SimpleNamespace(**{k:frozenset(map(tuple,row[k])) for k in ('floor','occupied','fine_occupied')})
        position=np.asarray(row['position_map_xy_m']);goal=np.asarray(row['goal_map_xy_m'])
        route=refine_proposal(row['proposal'],snapshot.floor,snapshot.occupied)
        expected=without_time(cached_fine_goal_route(snapshot,position,goal,route))
        trials=[]
        for name in ('reference','cached_graph','cached_graph','reference'):
            function=cached_fine_goal_route if name=='reference' else fine_goal_route
            cached_graph_geometry.cache_clear();_cached_clearance.cache_clear()
            search_graph.cache_clear();floor_index.cache_clear()
            elapsed=[]
            for _ in range(6):
                start=time.perf_counter_ns();result=function(snapshot,position,goal,route)
                elapsed.append((time.perf_counter_ns()-start)/1e6)
                assert without_time(result)==expected
            trials.append(dict(implementation=name,cold_ms=elapsed[0],warm_ms=elapsed[1:]))
        result=dict(frame=row['frame'],floor_cells=len(snapshot.floor),
            status=expected['status'],outputs_match_except_elapsed_time=True,trials=trials,
            warm_median_ms={name:float(np.median([v for t in trials if t['implementation']==name for v in t['warm_ms']]))
                for name in ('reference','cached_graph')})
        rows.append(result);print(json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)
    output.write_text(json.dumps(dict(schema='cached_fine_connectivity_comparison.v1',rows=rows,
        compared_calls=len(rows)*24,all_outputs_match_except_elapsed_time=True,
        exact_maps_and_seed_and_target_key_graph_cache=True,current_connectors_rechecked=True,
        native_state_used=False,full_controller_deadline_improvement_proven=False,
        alternative_navigation_outcome_proven=False),indent=2)+'\n')


if __name__=='__main__':main()
