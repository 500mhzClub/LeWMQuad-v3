"""Actual native trajectories for the completed fixed coverage comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from scripts import run_go2_balanced_start_goal_pilot_development as pilot


def main():
    report=json.loads(Path('docs/go2_balanced_start_goal_pilot_result_2026-09-17.json').read_text())
    assert report['status']=='COMPLETE'
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    styles={'old_action':('#176baf','--','Original-data predictor'),
            'mixed_action':('#278342','-','Supplemented predictor'),
            'old_no_future_action':('#b35825',':','Action-blind policy')}
    for task,ax in enumerate(axes.flat):
        scene=pilot.CASES[3*task][0];spec=pilot.previous.previous.specification(scene)
        box=spec['geometry']['wall_boxes'][0];x,y,_=box['centre_xyz'];w,h,_=box['size_xyz']
        ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,color='#cccccc',label='Obstacle'))
        for case in range(task*3,task*3+3):
            row=report['cases'][case];root=pilot.root(case)/f'case_{case:02d}'
            with np.load(root/'physics_trace.npz',allow_pickle=False) as a:xy=a['base_pose_world'][:,:2]
            colour,line,name=styles[row['arm']]
            label=f"{name}: {row['final_xy_error_m']*100:.2f} cm / {row['final_yaw_error_deg']:.1f}°"
            ax.plot(xy[:,0],xy[:,1],color=colour,ls=line,lw=1.8,label=label)
            ax.scatter(*xy[-1],color=colour,marker='X' if row['disallowed_contact'] else 'o',s=45,zorder=5)
            goal=np.asarray(json.loads((root/'result.json').read_text())['goal_pose_evaluator_only'])
        ax.scatter(*goal[:2],marker='*',s=150,color='black',label='Goal',zorder=6)
        ax.add_patch(Circle(goal[:2],.03,fill=False,color='black',ls='--',label='3 cm position tolerance'))
        ax.scatter(0,0,marker='+',color='black',s=60)
        ax.set(xlim=(-.22,.9),ylim=(-.6,.6),xlabel='World x (m)',ylabel='World y (m)',title=scene)
        ax.set_aspect('equal');ax.grid(alpha=.18);ax.legend(fontsize=7,loc='lower left')
    fig.suptitle('Balanced start coverage: 2/4 final arrivals versus 0/4 matched original-data continuation\nFour exposed local tasks; robot-centre paths; heading tolerance 5°; X marks contact',fontsize=12)
    path=Path('docs/go2_balanced_start_goal_pilot_2026-09-17.png')
    fig.savefig(path,dpi=150);print(path)


if __name__=='__main__':main()
