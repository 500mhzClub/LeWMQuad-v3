"""Plot a verified native round trip; simulator truth is used only by this evaluator."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from lewm.physical_execution_development import rotation_xyzw


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    args=parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary development artifact basename required')
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')/args.root_name
    targets=[root/f'verified_round_trip.{extension}' for extension in ('png','svg')]
    if any(p.exists() for p in targets):raise ValueError('preserve existing figures')
    evaluation=json.loads((root/'continuous_native_arrival_evaluation.json').read_text())
    if not evaluation['round_trip_arrival_checks_passed']:raise ValueError('independent verified round trip required')
    launch=json.loads((root/'launch.json').read_text())
    metadata=json.loads((root/'native/in_memory_camera_observations.json').read_text())
    frames={r['frame']:r for r in metadata['frames']}
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:physics=data['base_pose_world']
    origin=physics[frames[0]['physical_sample_index']];R0=rotation_xyzw(origin[3:])
    goal=origin[:3]+R0@np.r_[launch['public_mission']['goal_initial_body_xy_m'],0.]
    walls=json.loads((root/'native/camera_setup_identity.json').read_text())['environment']['physical_geometries']
    outbound,returned=evaluation['arrivals']
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.6),sharex=True,sharey=True)
    for ax,(first,last,title,color,arrival) in zip(axes,(
            (0,outbound['frame'],'Outbound','#1672b8',outbound),
            (outbound['frame'],returned['frame'],'Return','#c75624',returned))):
        for wall in walls:
            if wall['geom_type']!='BOX':continue
            half=np.asarray(wall['data'][:2])/2
            local=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half
            quat=np.asarray(wall['quaternion_world_wxyz']);R=rotation_xyzw(quat[[1,2,3,0]])
            points=np.c_[local,np.zeros(4)]@R.T+np.asarray(wall['position_world_m'])
            ax.add_patch(Polygon(points[:,:2],facecolor='#49515b',edgecolor='none'))
        xy=physics[[frames[f]['physical_sample_index'] for f in range(first,last+1)],:2]
        ax.plot(xy[:,0],xy[:,1],color=color,lw=2,label='Native robot trajectory')
        ax.scatter(*origin[:2],s=65,marker='o',color='#126144',edgecolor='white',zorder=4,label='Home')
        ax.scatter(*goal[:2],s=130,marker='*',color='#e1a322',edgecolor='#694c0b',zorder=4,label='Goal')
        ax.set_title(f'{title}: {(last-first)/10:.1f} s\nVerified dwell ≤ {arrival["native_maximum_distance_m"]*1000:.1f} mm',fontsize=11)
        ax.set_xlabel('World x (m)');ax.set_aspect('equal');ax.grid(alpha=.15)
        ax.margins(.06)
    axes[0].set_ylabel('World y (m)');axes[1].legend(loc='upper right',fontsize=8)
    fig.suptitle(f'Verified continuous round trip — development layout {launch["layout_index"]:02d}',fontsize=14)
    fig.text(.5,.025,'Ideal-sensor simulation · zero disallowed contacts · native truth used only for evaluation',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.93))
    for path in targets:fig.savefig(path,dpi=170)
    plt.close(fig)
    print(json.dumps({'figures':[str(p) for p in targets]}))


if __name__=='__main__':main()
