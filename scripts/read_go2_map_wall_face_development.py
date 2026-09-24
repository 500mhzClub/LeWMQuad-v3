"""Show the observed wall face and its later reverse-face observations."""
import json
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from scripts.replay_go2_no_early_release_map_entry_development import ROOT


def main():
    read=lambda name:json.loads((ROOT/name).read_text())
    replay=read('map_entry_replay_v1.json')
    walls=read('launch.json')['fresh_layout_inventory']['layouts'][1]['geometry']['wall_boxes']
    initial_index=read('native/in_memory_camera_observations.json')['frames'][0]['physical_sample_index']
    with np.load(ROOT/'native/physics_trace.npz',allow_pickle=False) as data:
        initial=data['base_pose_world'][initial_index].copy()
    R0=rotation_xyzw(initial[3:]);B=np.array(replay['snapshots']['572']['map_from_initial'])
    wall=next(w for w in walls if w['wall_id']=='independent_round_trip_wall_0_-1_0')
    centre=np.array(wall['centre_xyz']);half=np.array(wall['size_xyz'])*.5
    corners=np.array([centre+np.array([i,j,0])*half for i,j in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    mapped=((corners-initial[:3])@R0)@B.T
    rows=[]
    for frame in (80,400,520,556,564,572,596,624,628,632):
        cells=np.array(replay['snapshots'][str(frame)]['fine_occupied'])
        region=cells[(cells[:,0]>=165)&(cells[:,0]<=215)&(cells[:,1]>=-80)&(cells[:,1]<=-55)]
        rows.append(dict(map_frame=frame,cells_in_corner_region=len(region),
            occupied_y_indices=np.unique(region[:,1]).tolist(),
            maximum_x_index_by_y={str(int(y)):int(region[region[:,1]==y,0].max()) for y in np.unique(region[:,1])}))
    report=dict(schema='map_wall_face_readout.v1',wall=wall,
        wall_horizontal_corners_map_m=mapped[:,:2].tolist(),
        corner_region_cell_indices=dict(x=[165,215],y=[-80,-55]),rows=rows,
        native_geometry_evaluator_only=True,stored_cell_coordinates_are_recorded_reconstruction=True,
        wall_thickness_is_not_controller_input=True,
        result='The older nearby cells represent one wall face; the reverse face near the wall end enters the stored map only later.')
    with (ROOT/'map_wall_face_readout_v1.json').open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    fig,axes=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    for ax,map_frame,plan_frame in zip(axes,(572,632),(576,636)):
        snapshot=replay['snapshots'][str(map_frame)]
        position=np.array(next(r['position_map_m'] for r in replay['plan_comparisons'] if r['frame']==plan_frame))
        floor=np.array(snapshot['floor']);fine=np.array(snapshot['fine_occupied'])
        ax.scatter((floor[:,0]+.5)*.05,(floor[:,1]+.5)*.05,c='#d4ead4',s=55,marker='s',label='Observed floor')
        ax.add_patch(Polygon(mapped[:,:2],facecolor='#d0d0d0',edgecolor='#666666',label='Physical wall (evaluation only)'))
        ax.scatter((fine[:,0]+.5)*.01,(fine[:,1]+.5)*.01,c='#b84028',s=6,marker='s',label='Stored obstacle cells')
        ax.scatter(*position[:2],c='black',s=22,label='Estimated base position')
        ax.add_patch(Circle(position[:2],.45,fill=False,edgecolor='black',linestyle='--'))
        ax.set_xlim(1.5,2.8);ax.set_ylim(-1.6,-.3);ax.set_aspect('equal')
        ax.set_xlabel('Map x (m)');ax.set_ylabel('Map y (m)')
        ax.set_title(f'Map {map_frame}, planning pose {plan_frame}')
    axes[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('A stored surface does not describe the hidden side of a wall\nDashed circle: 0.45 m nominal footprint; white cells have no retained floor observation.',fontsize=11)
    for ext in ('png','svg'):
        fig.savefig(ROOT/f'map_wall_face_readout_v1.{ext}',dpi=160)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
