"""Plot actual robot-centre trajectories, target poses and the fresh outcomes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from scripts import run_go2_fresh_visual_goal_comparison_development as pilot


def main():
    report=json.loads(Path('docs/go2_fresh_visual_goal_comparison_result_2026-09-17.json').read_text())
    assert report['status']=='COMPLETE'
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    colours={'world_model':'#176baf','direct_feedback':'#db6927'}
    for task,ax in enumerate(axes.flat):
        name=pilot.TASKS[task][0];spec=pilot.specification(name)
        panel=spec['geometry']['wall_boxes'][0];x,y,_=panel['centre_xyz'];w,h,_=panel['size_xyz']
        ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,color='#bbbbbb',label='Obstacle'))
        for case in (2*task,2*task+1):
            row=report['cases'][case];directory=pilot.OUTPUT/f'case_{case:02d}'
            with np.load(directory/'physics_trace.npz',allow_pickle=False) as trace:
                pose=trace['base_pose_world'];xy=pose[:, :2]
            colour=colours[row['controller']]
            label=('World model' if case%2==0 else 'Direct feedback')+f" ({row['final_xy_error_m']*100:.2f} cm, {row['final_yaw_error_deg']:.1f}° final)"
            ax.plot(xy[:,0],xy[:,1],color=colour,lw=1.8,label=label)
            ax.scatter(*xy[-1],color=colour,marker='X' if row['disallowed_contact'] else 'o',s=55,zorder=4)
            result=json.loads((directory/'result.json').read_text())
            goal=np.asarray(result['goal_pose_evaluator_only'])
        ax.scatter(goal[0],goal[1],marker='*',s=140,color='#278342',label='Goal centre',zorder=5)
        ax.add_patch(Circle(goal[:2],.03,fill=False,color='#278342',ls='--',label='3 cm position tolerance'))
        ax.scatter(0,0,color='black',marker='+',s=60)
        ax.set(xlim=(-.22,.9),ylim=(-.6,.6),xlabel='World x (m)',ylabel='World y (m)',title=f'{name}: {pilot.TASKS[task][1]}')
        ax.set_aspect('equal');ax.grid(alpha=.18);ax.legend(fontsize=7,loc='lower left')
    fig.suptitle('Fresh local tasks — 0/4 final arrivals for either controller\nRobot-centre paths; arrival also requires heading within 5°; X marks contact',fontsize=13)
    path=Path('docs/go2_fresh_visual_goal_comparison_2026-09-17.png')
    fig.savefig(path,dpi=150);print(path)


if __name__=='__main__':main()
