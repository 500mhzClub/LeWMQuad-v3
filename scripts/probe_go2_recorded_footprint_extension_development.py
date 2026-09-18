"""Test observed-floor extension on saved forecasts, without executing alternatives."""
import itertools
import json
import time

import cv2
import numpy as np
import torch

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.observed_geometry_refinement_development import segment_cell_distances
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.replay_go2_no_early_release_map_entry_development import ROOT, RecordedCurrentPlaneMap


def footprint_extension(prediction, position, rotation, floor):
    """Unknown 5-cm cells swept by a 0.48-m disk beyond current/hold coverage.

    Retaining current and hold-path unknown cells is a development comparison
    baseline, not a declaration that those cells are free or safe.
    """
    paths = np.asarray(position)[:2]+np.concatenate((np.zeros((6,1,2)),
        np.asarray(prediction)[:,:,:2]),axis=1)@np.asarray(rotation)[:2,:2].T
    low = np.floor((paths.min(axis=(0,1))-.48)/.05).astype(int)
    high = np.floor((paths.max(axis=(0,1))+.48)/.05).astype(int)
    if np.any(low < -100) or np.any(high >= 100):
        raise ValueError('this recorded-state probe requires footprints inside the existing map bounds')
    cells = np.array(list(itertools.product(range(low[0],high[0]+1),range(low[1],high[1]+1))))
    unknown = np.array([tuple(c) not in floor for c in cells])
    current = (segment_cell_distances(paths[0,0],paths[0,0],cells)<=.48)&unknown
    masks = [np.logical_or.reduce([segment_cell_distances(a,b,cells)<=.48
        for a,b in zip(path,path[1:])])&unknown for path in paths]
    baseline = current|masks[ACTIONS.index('hold')]
    return dict(radius_m=.48,current_unknown_cells=int(current.sum()),
        baseline_unknown_cells=int(baseline.sum()),
        candidates=[dict(action=action,new_unknown_cells=int((mask&~baseline).sum()),
            added_cells=cells[mask&~baseline].tolist()) for action,mask in zip(ACTIONS,masks)])


def main():
    output = ROOT/'recorded_footprint_extension_all_plans_v1.json'
    if output.exists():raise ValueError('preserve completed probe')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1);configure()
    read = lambda name:json.loads((ROOT/name).read_text())
    poses = {r['frame']:r['registered_pose'] for r in read('poses.json')}
    plans = [r for r in read('planning.json') if 'selection' in r and r['frame']<=640]
    frames = sorted({r['frame'] for r in read('stage_events.json') if r['stage']=='mapping' and r['frame']<=636})
    reader=NoisyPublicReplay(ROOT/'native');mapper=RecordedCurrentPlaneMap();snapshots={};began=time.monotonic()
    for frame in frames:
        policy,depth,_,_,auxiliary,now=reader.packet(frame)
        snapshots[frame]=mapper.update(policy,depth,poses[frame],auxiliary_depth=auxiliary,measured_ns=now)
    rows=[]
    for plan in plans:
        snapshot=snapshots[plan['map_frame']];B=np.asarray(snapshot.map_from_initial)
        pose=poses[plan['frame']];q=B@np.asarray(pose['position_initial_body_m'])
        rotation=B@np.asarray(pose['rotation_initial_body_from_current_body'])
        receipt=footprint_extension(plan['motion_correction']['applied_prediction_after_yaw_ablation'],q,rotation,snapshot.floor)
        s=plan['selection'];extra={r['action']:r['new_unknown_cells'] for r in receipt['candidates']}
        stopping={r['action']:r['projection_clear'] for r in s['planned_stopping_projection']['candidates']}
        utilities={r['action']:r['utility_m'] for r in s.get('scan_utilities',s['candidates'])}
        eligible=[r['action'] for r in s['memory_forecast_candidates'] if
            r['nominal_predicted_path_clear'] and stopping[r['action']] and
            r['action'] in utilities and extra[r['action']]==0]
        changed=extra[plan['action']]>0
        alternative=(max(eligible,key=lambda a:utilities[a]) if eligible else 'hold') if changed else plan['action']
        rows.append(dict(frame=plan['frame'],map_frame=plan['map_frame'],original_action=plan['action'],
            original_action_introduces_extra_unknown=changed,existing_clear_alternatives=eligible,
            alternative_action=alternative,receipt=receipt))
    changed=[r for r in rows if r['original_action_introduces_extra_unknown']]
    result=dict(schema='recorded_footprint_extension_all_plans.v1',plans=len(rows),
        changed_selections=len(changed),changed_frames=[r['frame'] for r in changed],
        first_changed=changed[0] if changed else None,
        changed_states_with_no_existing_clear_alternative=sum(not r['existing_clear_alternatives'] for r in changed),
        wall_seconds=time.monotonic()-began,rows=rows,
        native_state_and_wall_geometry_used=False,current_or_hold_unknown_not_declared_free=True,
        existing_nominal_reserve_and_stopping_checks_retained=True,
        runtime_latch_transitions_not_replayed=True,new_closed_loop_navigation_executed=False,
        alternative_safety_or_navigation_success_proven=False)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','first_changed')},indent=2),flush=True)
    print('FIRST_CHANGED',json.dumps(result['first_changed']),flush=True)


if __name__=='__main__':main()
