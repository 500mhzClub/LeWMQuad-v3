"""Compare current-plane/raw-pixel coverage against the exact recorded map prefix."""
import hashlib
import json
import time
import numpy as np
from lewm.current_plane_floor_coverage_development import CurrentPlaneCoverageMap
from lewm.fine_stored_obstacle_routing_development import proposer
from lewm.two_cm_floor_extent_development import configure
from scripts.probe_go2_noisy_routing_floor_development import LocalCoverageMap,recorded_pose
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE


class RecordedPlaneMap(CurrentPlaneCoverageMap):
    retain_fine_obstacles=True
    _read_pose=staticmethod(recorded_pose)


def main():
    root=BASE/'go2_live_gyro_height_floor_noise_2mm_native_layout02_4800_v1_attempt_001'
    output=root/'current_plane_floor_coverage_prefix_probe_v1.json'
    if output.exists():raise ValueError('preserve completed probe')
    configure()
    prefix=json.loads((root/'union_floor_coverage_prefix_probe_v1.json').read_text())
    plans=json.loads((root/'planning.json').read_text())
    witnesses={p['map_frame']:p['selection']['routing_memory_scope'] for p in plans if 'selection' in p}
    poses={p['frame']:p['registered_pose'] for p in json.loads((root/'poses.json').read_text())}
    launch=json.loads((root/'launch.json').read_text());reader=NoisyPublicReplay(root/'native')
    baseline=LocalCoverageMap();candidate=RecordedPlaneMap();rows=[];checked=[];started=time.monotonic()
    for item in prefix['frames']:
        frame=item['frame'];p,d,fast,rgb,aux,now=reader.packet(frame)
        args=dict(auxiliary_depth=aux,measured_ns=now)
        a=baseline.update(p,d,poses[frame],**args);b=candidate.update(p,d,poses[frame],**args)
        assert a.floor_height==b.floor_height and a.occupied==b.occupied and a.fine_occupied==b.fine_occupied
        if frame in witnesses:
            w=witnesses[frame]
            assert len(a.floor)==w['retained_floor_cells'] and len(a.fine_occupied)==w['retained_fine_obstacle_cells'],frame
            checked.append(frame)
        rows.append(dict(frame=frame,baseline_floor_cells=len(a.floor),candidate_floor_cells=len(b.floor),
            current_paired_plane=candidate.last_floor_plane,
            candidate_primary_floor_cells=b.primary_current_floor_cells,
            candidate_auxiliary_floor_cells=b.auxiliary_current_floor_cells))
    goal=np.asarray(a.map_from_initial)@np.r_[launch['public_mission']['goal_initial_body_xy_m'],0.]
    excluded={tuple(c) for c in prefix['excluded_cells']}
    variants={n:dict(floor_cells=sorted(s.floor),route=proposer(s)(s.floor,s.occupied,
        s.position_map[:2],goal[:2],excluded_frontiers=excluded)) for n,s in [('baseline',a),('current_plane',b)]}
    report=dict(mapping_frames=len(rows),exact_saved_count_witnesses=checked,frames=rows,variants=variants,
        current_plane_available=sum(r['current_paired_plane']['available'] for r in rows),
        original_floor_height_and_raw_obstacles_unchanged=True,raw_validity_and_all_pixel_height_check_preserved=True,
        pixel_mesh_orientation_classifier_replaced=True,fixed_recorded_public_poses=True,
        actual_delivered_noisy_packets_verified=True,full_pose_acceptance_replayed=False,
        full_controller_state_replayed=False,counterfactual_navigation_executed=False,
        native_physics_used=False,wall_seconds=time.monotonic()-started,
        source_sha256={p:hashlib.sha256(open(p,'rb').read()).hexdigest() for p in
            (__file__,'lewm/current_plane_floor_coverage_development.py')})
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(mapping_frames=len(rows),exact_witnesses=len(checked),
        current_plane_available=report['current_plane_available'],
        routes={n:v['route'] for n,v in variants.items()})),flush=True)


if __name__=='__main__':main()
