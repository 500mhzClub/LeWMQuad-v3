"""Compare the same stored obstacle points at 5 cm and 1 cm resolution."""
from pathlib import Path
import json
import cv2
import numpy as np
from scripts.diagnose_alignment_route_switches_development import RecordedPoseMap,saved_pose
from scripts.in_memory_public_replay_development import PublicReplay
from lewm.causal_depth_observation_development import body_points
from lewm.auxiliary_downward45_depth_observation_development import body_points as auxiliary_points


def main():
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_continuous_connector_optional_plane_native_layout00_v1_attempt_001')
    output=root/'stored_obstacle_resolution_diagnostic.json'
    if output.exists():raise ValueError('preserve diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    reader=PublicReplay(root/'native');mapper=RecordedPoseMap()
    poses={r['frame']:r['registered_pose'] for r in json.loads((root/'poses.json').read_text())}
    clouds=[];owners=[]
    for frame in range(0,105,4):
        policy,depth,_,_,aux,now=reader.packet(frame)
        snapshot=mapper.update(policy,depth,poses[frame],auxiliary_depth=aux,measured_ns=now)
        p,R,_=saved_pose(poses[frame]);B=mapper.B;Q,q=B@R,B@p
        for auxiliary,packet in ((False,depth),(True,aux)):
            cloud=(auxiliary_points if auxiliary else body_points)(packet,policy,now_ns=now,stride=4)
            points=cloud['points_body_m'][cloud['valid']]
            mapped=((points@R.T+p)@B.T) if auxiliary else (points@Q.T+q)
            above=mapped[(mapped[:,2]>mapper.floor_height+.03)&(mapped[:,2]<mapper.floor_height+.65)]
            above=above[(above[:,:2]>=-5.).all(axis=1)&(above[:,:2]<5.).all(axis=1)]
            clouds.append(above);owners.extend([(frame,'auxiliary' if auxiliary else 'primary')]*len(above))
    points=np.concatenate(clouds);position=(mapper.B@np.asarray(poses[108]['position_initial_body_m']))[:2]
    pointdist=np.linalg.norm(points[:,:2]-position,axis=1);nearest=int(pointdist.argmin())
    quantized={}
    for cell_m in [.05,.01]:
        keys=np.unique(np.floor(points[:,:2]/cell_m).astype(int),axis=0)
        if cell_m==.05:assert {tuple(k) for k in keys}==snapshot.occupied
        distances=np.linalg.norm(np.maximum(np.maximum(keys*cell_m-position,position-(keys+1)*cell_m),0.),axis=1)
        index=int(distances.argmin());key=keys[index]
        contained=(np.floor(points[:,:2]/cell_m).astype(int)==key).all(axis=1)
        quantized[str(cell_m)]=dict(cells=len(keys),minimum_cell_square_distance_m=float(distances[index]),
            nearest_cell=key.tolist(),points_in_nearest_cell=int(contained.sum()),
            contained_point_minimum_distance_m=float(pointdist[contained].min()))
    report=dict(map_frame=104,pose_frame=108,point_count=len(points),position_map_xy_m=position.tolist(),
        nominal_radius_m=.45,minimum_stored_point_distance_m=float(pointdist[nearest]),
        nearest_point=points[nearest].tolist(),nearest_point_source=owners[nearest],quantized=quantized,
        coarse_occupied_cells_reconstructed_exactly=True,native_state_read=False,
        point_samples_are_not_continuous_surface_coverage=True)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
