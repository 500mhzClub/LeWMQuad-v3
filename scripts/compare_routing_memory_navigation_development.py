"""Compare the fixed routing-memory arms after native arrival evaluation."""
import argparse
import json

from scripts.compare_continuous_navigation_arms_development import BASE, read, summarize


def scope_evidence(root, scope):
    plans = [p for p in read(root, 'planning.json') if 'selection' in p]
    if not plans:
        raise ValueError('selected plans required')
    rows = [p['selection']['routing_memory_scope'] for p in plans]
    for row in rows:
        if row['condition'] != scope or not row['accumulated_action_clearance_preserved']:
            raise ValueError('routing treatment or retained clearance differs')
        for kind in ('floor', 'fine_obstacle'):
            routed, retained = (row[f'{prefix}_{kind}_cells'] for prefix in ('routing', 'retained'))
            if routed > retained or (scope == 'persistent' and routed != retained):
                raise ValueError('routing counts contradict assigned scope')
    return dict(selected_plans=len(rows), all_assigned_scopes_present=True,
        accumulated_action_clearance_preserved=True,
        plans_with_reduced_floor=sum(r['routing_floor_cells'] < r['retained_floor_cells'] for r in rows),
        plans_with_reduced_fine_obstacles=sum(r['routing_fine_obstacle_cells'] < r['retained_fine_obstacle_cells'] for r in rows),
        cell_ranges={k: [min(r[k] for r in rows), max(r[k] for r in rows)]
            for k in ('routing_floor_cells', 'retained_floor_cells',
                      'routing_fine_obstacle_cells', 'retained_fine_obstacle_cells')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    i = args.layout_index
    scopes = ('persistent', 'latest_mapped_pair')
    roots = {s: BASE/f'go2_routing_memory_{s}_native_layout{i:02d}_4800_v1_attempt_001' for s in scopes}
    launches = {s: read(r, 'launch.json') for s, r in roots.items()}
    a, b = (launches[s] for s in scopes)
    differences = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    if set(differences) - {'owner', 'routing_memory_scope', 'comparison_condition'}:
        raise ValueError(f'unmatched launch settings: {differences}')
    for s, launch in launches.items():
        if launch['routing_memory_scope'] != s or launch['layout_index'] != i:
            raise ValueError('assigned layout and scope required')
    report = dict(layout_index=i, comparison='persistent_versus_latest_mapped_pair_routing',
        differing_launch_fields=differences, all_other_launch_fields_and_source_hashes_equal=True,
        conditions={s: summarize(r) for s, r in roots.items()},
        routing_scope_evidence={s: scope_evidence(r, s) for s, r in roots.items()},
        development_layout_revisit=True, fully_memoryless_comparison=False,
        accumulated_action_clearance_and_other_controller_state_retained=True,
        model_internal_memory_ablation=False, jepa_specific_advantage_established=False,
        hardware_validated=False)
    output = BASE/f'go2_routing_memory_comparison_layout{i:02d}_v1_attempt_001'
    output.mkdir()
    with (output/'result.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(dict(output=str(output), scope_evidence=report['routing_scope_evidence'])))


if __name__ == '__main__':
    main()
