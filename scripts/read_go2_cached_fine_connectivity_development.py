"""Evaluate the complete cache trial and contrast its timing with its reference."""
import json
from collections import Counter
import numpy as np

from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_cached_fine_connectivity_development as run
from scripts import read_go2_current_position_coverage_view_development as views


def main():
    root=run.BASE/run.ROOT;output=root/'cached_fine_connectivity_readout_v1.json'
    if output.exists():raise ValueError('preserve completed readout')
    bind(views.main,BASE=run.BASE,ROOT=run.ROOT,PLAN=run.PLAN)()
    read=lambda name:json.loads((root/name).read_text())
    plans=[p for p in read('planning.json') if 'selection' in p]
    profile=read('live_planning_profile.json')
    result=dict(schema='cached_fine_connectivity_readout.v1',
        navigation=read('short_pulse_navigation_evaluation_v1.json'),
        coverage=read('current_position_coverage_view_readout_v1.json'),
        planning_latency_ms=dict(zip(('median','p95','maximum'),np.percentile(
            [(p['completed_ns']-p['measured_ns'])/1e6 for p in plans],[50,95,100]).tolist())),
        component_wall_ms={name:dict(zip(('median','p95','maximum'),np.percentile(
            [p['components'][name]['wall_ns']/1e6 for p in profile if name in p['components']],
            [50,95,100]).tolist())) for name in ('route','model_forward','action_selection')},
        dispatch_reasons=dict(Counter(r['reason'] for r in read('requests.json'))),
        reference_navigation=json.loads((run.REFERENCE/'short_pulse_navigation_evaluation_v1.json').read_text()),
        asynchronous_trajectories_not_matched=True,causal_navigation_improvement_proven=False,
        fresh_layout_replication=False,hardware_validated=False)
    # Full per-plan and event details remain in the named coverage receipt.
    result['coverage']={k:v for k,v in result['coverage'].items() if k not in ('rows','events')}
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('coverage','dispatch_reasons')},indent=2))


if __name__=='__main__':main()
