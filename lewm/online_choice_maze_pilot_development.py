"""Fresh, fixed online-choice layouts; old dataset generator remains immutable."""
import copy

import numpy as np

from lewm.counterfactual_maze_development import corpus as previous_corpus,topology_identity
from lewm.multijunction_routes_development import connection

METHODS=('always_stop','supervised_rollout','jepa')
INTENTS=(('forward',(.8,0.)),('left',(0.,.8)),('right',(0.,-.8)))


def layout_spec(index):
    if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<8:
        raise ValueError('fixed online pilot has eight layouts')
    seed=2026091400+index; rng=np.random.default_rng(seed)
    cells=[(x,y) for x in range(-1,3) for y in range(-1,3)]
    source,junction=(-1,0),(0,0)
    exits=(((0,1),(0,-1)),((1,0),(0,1)),((1,0),(0,-1)),((1,0),(0,1),(0,-1)))[index%4]
    links={connection(source,junction)} | {connection(junction,end) for end in exits}
    parent={cell:cell for cell in cells}
    def root(cell):
        while parent[cell]!=cell: cell=parent[cell]
        return cell
    def merge(a,b): parent[root(a)]=root(b)
    for a,b in sorted(links): merge(a,b)
    candidates=[]
    for cell in cells:
        for dx,dy in ((1,0),(0,1)):
            other=(cell[0]+dx,cell[1]+dy); edge=connection(cell,other)
            if other not in parent: continue
            if source in edge and edge!=connection(source,junction): continue
            if junction in edge and edge not in links: continue
            candidates.append(edge)
    rng.shuffle(candidates)
    for a,b in candidates:
        if root(a)!=root(b): links.add(connection(a,b)); merge(a,b)
    for edge in candidates:
        if edge not in links and rng.random()<.12: links.add(edge)
    if len({root(c) for c in cells})!=1: raise ValueError('disconnected fixed pilot topology')
    width=float(rng.uniform(.95,1.25)); pitch=width+.08
    walls={}
    for cell in cells:
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            if connection(cell,(cell[0]+dx,cell[1]+dy)) in links: continue
            midpoint=(2*cell[0]+dx,2*cell[1]+dy); key=(midpoint,abs(dx))
            walls[key]={'wall_id':f'wall_{midpoint[0]}_{midpoint[1]}_{abs(dx)}',
                'centre_xyz':[midpoint[0]*pitch/2,midpoint[1]*pitch/2,.3],
                'size_xyz':[.08,pitch+.08,.6] if dx else [pitch+.08,.08,.6],
                'yaw_rad':0.,'material_id':'NEUTRAL_WALL'}
    def region(cell):
        center=np.array(cell)*pitch
        return [(center+np.array(offset)*width/2).tolist() for offset in ((-1,-1),(1,-1),(1,1),(-1,1))]
    geometry={'spawn_se2_world':[-pitch,float(rng.uniform(-.07,.07)),float(rng.uniform(-.12,.12))],
        'wall_boxes':[walls[k] for k in sorted(walls)],
        'source_node':{'node_id':'prefix_source','centre_world':[-pitch,0.],'boundary_polygon_world':region(source)},
        'target_node':{'node_id':'junction','centre_world':[0.,0.],'boundary_polygon_world':region(junction)},
        'selected_directed_edge':{'edge_id':'prefix-edge','source_node_id':'prefix_source','target_node_id':'junction',
            'opening_segment_world':[[-pitch/2,-width/2],[-pitch/2,width/2]],'opening_normal_world':[1.,0.],
            'edge_region_polygon_world':region(junction)},'competing_directed_edges':[],
        'teacher_route_polyline_world':[[-pitch,0.],[-pitch/2,0.],[0.,0.]]}
    return {'layout_id':f'online-choice-maze-development-v1-{index:02d}','layout_index':index,
        'data_role':'online_development','procedural_seed':seed,'topology_sha256_dihedral':topology_identity(links),
        'graph_connections':[[list(a),list(b)] for a,b in sorted(links)],'junction_exits':[list(e) for e in exits],
        'width_m':width,'cell_pitch_m':pitch,'geometry':geometry}


def corpus():
    layouts=[layout_spec(i) for i in range(8)]
    hashes={row['topology_sha256_dihedral'] for row in layouts}
    if len(hashes)!=8 or hashes & {row['topology_sha256_dihedral'] for row in previous_corpus()}:
        raise ValueError('fixed pilot topology overlap; no runtime resampling')
    return layouts


def trials():
    return [{**copy.deepcopy(layout),'scene_id':f'{layout["layout_id"]}-{intent}-{method}',
        'family':'ONLINE_CONDITIONAL_CHOICE_DEVELOPMENT','case_index':layout['layout_index'],'arm':'baseline',
        'method':method,'intent_name':intent,'intent_xy_body_start_m':list(goal)}
        for layout in corpus() for intent,goal in INTENTS for method in METHODS]


def realized_cost(labels,*,prefix_available,stop_reason,intent_xy):
    """Conservative all-trial utility; never impute unavailable post-stop motion."""
    if not prefix_available: return {'cost':10.,'kind':'prefix_failure','contact_by_4s':None,'progress_m':None}
    if len(labels)!=8 or labels[-1]['horizon_ns']!=4_000_000_000: raise ValueError('full declared horizon label population required')
    last=labels[-1]
    if last['contact_valid'] and last['contact_by_horizon']:
        return {'cost':10.,'kind':'contact_by_4s','contact_by_4s':True,'progress_m':None}
    if not last['motion_valid'] or not last['contact_valid']:
        return {'cost':10.,'kind':'noncontact_incomplete','contact_by_4s':None,'progress_m':None}
    goal=np.asarray(intent_xy,dtype=float); delta=np.asarray(last['delta_xy_yaw_start_body'],dtype=float)
    if goal.shape!=(2,) or delta.shape!=(3,) or not np.isfinite(goal).all() or not np.isfinite(delta).all():
        raise ValueError('invalid realized displacement/intent')
    distance=float(np.linalg.norm(delta[:2]-goal))
    # A failure during release is separately visible and is not called success
    # merely because an earlier 4-s endpoint was observed.
    cost=10. if stop_reason is not None else distance
    return {'cost':cost,'kind':'post_horizon_failure' if stop_reason is not None else 'observed',
        'contact_by_4s':False,'progress_m':float(np.linalg.norm(goal)-distance)}
