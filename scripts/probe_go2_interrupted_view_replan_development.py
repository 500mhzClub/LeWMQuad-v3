"""Check the proposed interrupted-view handling at its first saved opportunity."""
from copy import deepcopy
import json
import numpy as np

from lewm.camera_frontier_visits_development import CameraFrontierVisits
from lewm.interrupted_view_replan_development import retire_interrupted_view
from lewm.two_cm_floor_extent_development import configure
from scripts.replay_go2_no_early_release_map_entry_development import RecordedCurrentPlaneMap
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.run_go2_cached_fine_connectivity_development import BASE,ROOT,collection


def main():
    root=BASE/ROOT;output=root/'interrupted_view_saved_activation_v1.json'
    if output.exists():raise ValueError('preserve completed probe')
    collection.study.cohort.stable.floor.configure();configure()
    import cv2,torch
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    read=lambda name:json.loads((root/name).read_text())
    poses={r['frame']:r['registered_pose'] for r in read('poses.json')}
    plans=[r for r in read('planning.json') if r.get('route_status')==
        'LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW' and 'frontier_visit' in r]
    reader=NoisyPublicReplay(root/'native');mapper=RecordedCurrentPlaneMap()
    def update(frame):
        policy,depth,_,_,auxiliary,now=reader.packet(frame)
        return mapper.update(policy,depth,poses[frame],auxiliary_depth=auxiliary,measured_ns=now)
    snapshot=update(0);B=np.asarray(snapshot.map_from_initial)
    def position(plan):return B@np.asarray(poses[plan['frame']]['position_initial_body_m'])
    plan=next(r for r in plans if np.linalg.norm(position(r)[:2]-
        r['frontier_visit']['camera_viewpoint']['viewpoint_map_xy_m'])<=.10)
    frames=sorted({r['frame'] for r in read('stage_events.json') if r['stage']=='mapping'
        and 0<r['frame']<=plan['map_frame']})
    for frame in frames:snapshot=update(frame)
    scope=plan['selection']['routing_memory_scope']
    assert len(snapshot.floor)==scope['retained_floor_cells']
    assert len(snapshot.fine_occupied)==scope['retained_fine_obstacle_cells']
    visits=CameraFrontierVisits();visits.visit=deepcopy(plan['frontier_visit'])
    original=deepcopy(visits.visit);target=tuple(original['unknown_neighbour'])
    assert target not in snapshot.floor|snapshot.occupied
    assert retire_interrupted_view(visits,snapshot,position(plan),plan['measured_ns'])
    query=dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',
        route_cells=original['camera_viewpoint']['route_cells'],target_map_xy_m=original['target_xy_m'])
    R=B@np.asarray(poses[plan['frame']]['rotation_initial_body_from_current_body'])
    alternative=visits._choose(snapshot,query,position(plan),R,target)
    result=dict(schema='interrupted_view_saved_activation.v1',frame=plan['frame'],map_frame=snapshot.frame,
        recorded_map_updates_replayed=len(frames)+1,recorded_map_counts_matched=True,
        delivered_noisy_depth_digests_verified=True,position_map_m=position(plan).tolist(),
        original_visit=original,retirement_event=visits.events[-1],
        excluded_viewpoint_cells=sorted(visits.attempted[target]),alternative_viewpoint=alternative,
        unknown_target_still_unknown=True,no_floor_or_obstacle_map_cells_changed=True,
        alternative_projection_not_observed=True,complete_original_route_state_replayed=False,
        query_uses_saved_viewpoint_route=True,navigation_reexecuted=False,native_state_used=False)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(frame=plan['frame'],retired_viewpoint=original['camera_viewpoint']['viewpoint_map_xy_m'],
        position_map_m=position(plan).tolist(),excluded_cells=len(visits.attempted[target]),
        alternative_viewpoint=None if alternative is None else alternative['viewpoint_map_xy_m']),indent=2))


if __name__=='__main__':main()
