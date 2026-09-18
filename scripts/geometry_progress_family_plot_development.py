"""Native evaluator-only paths; caller authenticates exact source artifacts."""
import numpy as np
from lewm.geometry_progress_layout_family_development import layouts,geometry,ACTIONS,APPEARANCES
from lewm.geometry_progress_pilot_development import GOAL_BODY_XY
from lewm.physical_execution_development import rotation_xyzw


def plot_paths(reports,*,input_root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    colors=dict(zip(ACTIONS,['#777777','#171717','#0072B2','#D55E00','#56B4E9','#E69F00']))
    fig,axes=plt.subplots(4,2,figsize=(10,15),sharex=True,sharey=True)
    for (name,cell),ax in zip(layouts().items(),axes.flat,strict=True):
        for b in geometry(name)['wall_boxes']:
            x,y,_=b['centre_xyz'];dx,dy,_=b['size_xyz']
            ax.add_patch(Rectangle((x-dx/2,y-dy/2),dx,dy,facecolor='#dddddd',edgecolor='#555555'))
        rows=[r for r in reports if r['geometry']==name]
        if len(rows)!=12:raise ValueError('complete layout/appearance/action population required for plot')
        for r in rows:
            with np.load(input_root/r['trial']/'physics_trace.npz',allow_pickle=False) as z:poses=z['base_pose_world']
            if len(poses)<=899:continue
            path=poses[899:,:2];pose=poses[899];color=colors[r['action']]
            ax.plot(path[:,0],path[:,1],color=color,linewidth=1.4,
                linestyle='-' if r['appearance_seed']==APPEARANCES[0] else '--',alpha=.85)
            ax.scatter(*path[-1],color=color,marker='x' if r['outcome']['contact'] else 'o',s=28)
            target=pose[:3]+rotation_xyzw(pose[3:])@np.array([*GOAL_BODY_XY,0.])
            ax.scatter(*target[:2],marker='*',s=90,color='#009E73')
        ax.set(title=f"{cell['role'].replace('_',' ')} / {name.replace('_',' ')}\npanel x={cell['panel_x_mm']}mm, height={cell['panel_height_mm']}mm",
            xlim=(-.15,1.35),ylim=(-1.12,1.12),xlabel='Native world x (m)',ylabel='Native world y (m)',aspect='equal')
        ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([0],[0],color=colors[a],label=a.replace('_',' ')) for a in ACTIONS],
        loc='lower center',ncol=3)
    fig.suptitle('Recorded candidate paths on new local obstruction layouts\n'
        'Base centres only; crosses = contact stops, circles = other endpoints\n'
        'Solid/dashed = two appearances; fixed excitation, no learned policy execution',fontsize=11)
    fig.tight_layout(rect=(0,.045,1,.94))
    fig.savefig(output/'native_paths.png',dpi=150);fig.savefig(output/'native_paths.svg');plt.close(fig)
