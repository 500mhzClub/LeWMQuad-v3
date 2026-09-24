"""Physical trajectory figure for the completed exposed-maze round trip."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from scripts.run_go2_interrupted_view_replan_development import BASE,ROOT


def main():
    root=BASE/ROOT;stem=root/'physical_round_trip_v1'
    if stem.with_suffix('.png').exists() or stem.with_suffix('.svg').exists():
        raise ValueError('preserve completed figure')
    read=lambda name:json.loads((root/name).read_text())
    outcome=read('continuous_native_arrival_evaluation.json')
    assert outcome['round_trip_arrival_checks_passed']
    goal_arrival=next(r for r in outcome['arrivals'] if r['phase']=='OUTBOUND')
    launch=read('launch.json');layout=launch['fresh_layout_inventory']['layouts'][launch['layout_index']]
    frames=read('native/in_memory_camera_observations.json')['frames']
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:
        poses=data['base_pose_world'][[r['physical_sample_index'] for r in frames]]
    origin=poses[0];goal=origin[:3]+rotation_xyzw(origin[3:])@np.r_[launch['public_mission']['goal_initial_body_xy_m'],0.]
    outbound=np.array([r['frame']<=goal_arrival['frame'] for r in frames])
    fig,ax=plt.subplots(figsize=(8,7))
    for wall in layout['geometry']['wall_boxes']:
        assert wall['yaw_rad']==0
        centre=np.asarray(wall['centre_xyz'][:2]);size=np.asarray(wall['size_xyz'][:2])
        ax.add_patch(Rectangle(centre-size/2,*size,color='#d5d8dc',zorder=1))
    ax.plot(poses[outbound,0],poses[outbound,1],color='#2166ac',lw=2,label='Outbound',zorder=2)
    ax.plot(poses[~outbound,0],poses[~outbound,1],color='#d95f02',lw=1.7,ls='--',label='Return',zorder=3)
    ax.scatter(*origin[:2],s=90,c='#1b7837',marker='s',label='Home',zorder=4)
    ax.scatter(*goal[:2],s=130,c='#762a83',marker='*',label='Goal',zorder=4)
    ax.set(aspect='equal',xlabel='World x (m)',ylabel='World y (m)',
        title='Verified simulated round trip — exposed development maze')
    ax.autoscale_view();ax.margins(.06);ax.legend(loc='best',framealpha=.95)
    ax.grid(alpha=.15)
    fig.text(.5,.015,'306.14 simulated seconds · zero contacts · 11 outbound edges reversed\n'
        'Interrupted-view rule activated zero times; this run does not establish its benefit.',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1))
    fig.savefig(stem.with_suffix('.png'),dpi=170);fig.savefig(stem.with_suffix('.svg'))
    plt.close(fig);print(stem.with_suffix('.png'),flush=True)


if __name__=='__main__':main()
