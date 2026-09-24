"""Supplement completed dense navigation with physical backtracking evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path

from lewm import dense_world_model_maze_layouts_development as prospective
from lewm import sparse_corner_replication_layouts_development as exposed
from scripts.read_go2_interrupted_view_replan_development import physical_return_edges
from scripts import train_go2_horizon_dense_predictor_development as fit


def main(root_name):
    if Path(root_name).name != root_name or root_name.startswith('sealed'):
        raise ValueError('ordinary development artifact basename required')
    root = fit.OUTPUT.parent/root_name
    read = lambda name: json.loads((root/name).read_text())
    output = root/'dense_backtracking_readout.json'
    assert not output.exists() and not (root/'physical_return_corridor_readout_v1.json').exists()
    launch = read('launch.json')
    evaluated = read('dense_navigation_readout.json')
    layouts = prospective if launch.get('new_independent_development_layout') else exposed
    assert Path(launch['layout_source']).resolve() == Path(layouts.__file__).resolve()
    spec = layouts.specification(launch['layout_index'])
    physical = physical_return_edges(root, outcome=evaluated['physical'], layout=spec['evaluation_layout'])
    plans = [p for p in read('planning.json') if 'selection' in p]
    outbound = next((r for r in evaluated['physical']['arrivals']
        if r['phase']=='OUTBOUND' and r['arrival_checks_passed']), None)
    split = outbound['frame'] if outbound else float('inf')
    legs = {}
    for leg, selected in [('outbound', [p for p in plans if p['frame']<=split]),
            ('return', [p for p in plans if p['frame']>split])]:
        legs[leg] = dict(plans=len(selected),
            local_route_turn_memory_active_plans=sum(bool(p['selection'].get('visual_route_turn_memory', {}).get('active')) for p in selected),
            route_statuses=dict(Counter(p['route_status'] for p in selected)))
    result = dict(physical_backtracking=physical, legs=legs, model_assignment=launch['model_assignment'],
        motion_readout=launch.get('motion_readout'), native_state_and_layout_evaluator_only=True,
        memory_causal_advantage_established=False,
        interpretation='Reversing previously traversed corridors is physical backtracking; activation counts and round trips alone do not establish memory benefit.',
        source_sha256={p: fit.digest(p) for p in (__file__, 'scripts/read_go2_interrupted_view_replan_development.py')})
    fit.save(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    main(parser.parse_args().root_name)
