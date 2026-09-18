"""Post-hoc physical corridor reversals; no causal memory claim."""
import json,sys
from pathlib import Path
import numpy as np
base=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
fixed=json.loads(Path('docs/go2_nogil_navigation_replication_plan_2026-09-16.json').read_text())
for assignment in map(int,sys.argv[1:]):
 index,arm=fixed['assignments'][assignment-1]
 root=base/f'go2_nogil_replication_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
 read=lambda n:json.loads((root/n).read_text())
 outcome=read('continuous_native_arrival_evaluation.json')
 arrival=next((r for r in outcome['arrivals'] if r['phase']=='OUTBOUND' and r['arrival_checks_passed']),None)
 if arrival is None:
  print(assignment,'no verified outbound arrival; return-edge analysis not applicable');continue
 layout=read('launch.json')['fresh_layout_inventory']['layouts'][index]['evaluation_layout']
 pitch=layout['pitch_m'];cells={tuple(c) for c in layout['cells']}
 edges={tuple(sorted((tuple(a),tuple(b)))) for a,b in layout['edges']}
 frames=sorted(read('native/in_memory_camera_observations.json')['frames'],key=lambda r:r['frame'])
 with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:
  positions=data['base_pose_world'][[r['physical_sample_index'] for r in frames],:2]
 nearest=np.rint(positions/pitch).astype(int)
 transitions=[];outside=[];previous=None
 for record,cell,position in zip(frames,nearest,positions):
  cell=tuple(map(int,cell))
  if cell not in cells:
   outside.append(record['frame']);previous=None;continue
  if previous is not None and cell!=previous:
   transitions.append(dict(frame=record['frame'],source=previous,target=cell,
    graph_edge=tuple(sorted((previous,cell))) in edges,
    leg='outbound' if record['frame']<=arrival['frame'] else 'return'))
  previous=cell
 outbound={(tuple(r['source']),tuple(r['target'])) for r in transitions if r['leg']=='outbound' and r['graph_edge']}
 returning={(tuple(r['source']),tuple(r['target'])) for r in transitions if r['leg']=='return' and r['graph_edge']}
 reverse={(b,a) for a,b in outbound}&returning
 result=dict(schema='physical_return_corridor_readout.v1',assignment=assignment,arm=arm,layout_index=index,
  round_trip_verified=outcome['round_trip_arrival_checks_passed'],
  verified_outbound_frame=arrival['frame'],camera_rate_physics_samples=len(frames),
  cell_assignment='nearest world-grid cell centre at recorded camera frames',
  pitch_m=pitch,outside_known_grid_frames=outside,
  invalid_graph_transitions=sum(not r['graph_edge'] for r in transitions),
  outbound_unique_directed_edges=len(outbound),return_unique_directed_edges=len(returning),
  return_edges_reversing_observed_outbound_edges=len(reverse),
  reverse_edges=sorted(reverse),transitions=transitions,
  physical_backtracking_observed=bool(reverse) and not outside and all(r['graph_edge'] for r in transitions),
  native_state_and_maze_graph_evaluator_only=True,
  boundary_jitter_can_repeat_transitions=True,unique_edge_counts_used=True,
  exploratory_posthoc_readout=True,memory_causal_advantage_established=False)
 with (root/'physical_return_corridor_readout_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('transitions','reverse_edges','outside_known_grid_frames')}))
