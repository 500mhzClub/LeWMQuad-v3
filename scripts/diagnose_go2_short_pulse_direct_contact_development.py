"""Reconstruct the failed direct run's stored obstacle evidence from public depth."""
import json
import time
import cv2
import numpy as np
from lewm.current_plane_floor_coverage_development import CurrentPlaneFloorRoutingMap
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.two_cm_floor_extent_development import configure
from scripts.diagnose_alignment_route_switches_development import saved_pose
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.navigation_artifact_root_development import BASE

ROOT=BASE/'go2_short_pulse_navigation_direct_noise_2mm_native_layout00_4800_v1_attempt_001'
OUTPUT=BASE/'go2_short_pulse_direct_contact_map_replay_v1_attempt_001'


class RecordedMap(CurrentPlaneFloorRoutingMap):
    _read_pose=staticmethod(saved_pose)


def main():
    if OUTPUT.exists():raise ValueError('preserve completed or failed diagnostic')
    OUTPUT.mkdir();started=time.monotonic()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);configure()
    reader=NoisyPublicReplay(ROOT/'native');mapper=RecordedMap()
    read=lambda name:json.loads((ROOT/name).read_text())
    assert read('failure.json')['physical_stop']=='DISALLOWED_CONTACT'
    poses={p['frame']:p['registered_pose'] for p in read('poses.json')}
    plans=[p for p in read('planning.json') if 'selection' in p];plan=plans[-1]
    frames=sorted(e['frame'] for e in read('stage_events.json')
        if e['stage']=='mapping' and e['frame']<=plan['map_frame'])
    if len(frames)!=len(set(frames)) or frames!=list(range(0,plan['map_frame']+1,4)):
        raise ValueError('recorded mapping population differs from regular cadence')
    samples=[]
    for frame in frames:
        policy,depth,_,_,auxiliary,now=reader.packet(frame)
        snapshot=mapper.update(policy,depth,poses[frame],auxiliary_depth=auxiliary,measured_ns=now)
        if frame%400==0:
            row=dict(frame=frame,fine_cells=len(snapshot.fine_occupied),wall_s=time.monotonic()-started)
            samples.append(row);print('MAP_REPLAY',json.dumps(row),flush=True)
    B=np.asarray(snapshot.map_from_initial);p,R,_=saved_pose(poses[plan['frame']]);q=B@p
    actual=cached_clearance(snapshot.fine_occupied).minimum(q[:2],q[:2])
    expected=plan['lookahead']['start_clearance_m']
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    expected_count=plan['selection']['routing_memory_scope']['retained_fine_obstacle_cells']
    if len(snapshot.fine_occupied)!=expected_count:raise ValueError('stored obstacle count differs')
    cells=np.asarray(sorted(snapshot.fine_occupied),dtype=int)
    gaps=np.maximum(np.maximum(cells*.01-q[:2],q[:2]-(cells+1)*.01),0)
    nearest=int(np.linalg.norm(gaps,axis=1).argmin())
    with (OUTPUT/'stored_obstacles.npz').open('xb') as f:
        np.savez_compressed(f,cells=cells,map_from_initial=B,position_map=q,
            floor_height=snapshot.floor_height,planning_rotation_initial_from_body=R,
            floor_cells=np.asarray(sorted(snapshot.floor),dtype=int))
    result=dict(status='COMPLETE',root_name=ROOT.name,mapped_frames=len(frames),
        last_map_frame=snapshot.frame,plan_frame=plan['frame'],stored_fine_cells=len(cells),
        replayed_start_clearance_m=actual,recorded_start_clearance_m=expected,
        nearest_stored_cell=cells[nearest].tolist(),position_map=q.tolist(),
        floor_height=snapshot.floor_height,wall_s=time.monotonic()-started,
        progress=samples,public_depth_and_recorded_estimator_output_only=True,
        native_geometry_or_pose_read=False,live_noise_digests_verified=True,
        controller_unchanged=True)
    with (OUTPUT/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
