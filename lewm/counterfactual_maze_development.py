"""Fixed scene-role procedural corpus and executed-horizon labels for development."""
import copy
import hashlib
import json
import math
import numpy as np

from lewm.multijunction_routes_development import connection
from lewm.physical_execution_development import rotation_xyzw

ACTIONS=(('stop',(0.,0.,0.)),('forward',(.3,0.,0.)),('forward_left',(.2,0.,.5)),
         ('forward_right',(.2,0.,-.5)),('reverse',(-.2,0.,0.)))
HORIZONS_NS=tuple(range(500_000_000,4_000_000_001,500_000_000))


def topology_identity(links):
    variants=[]
    for reflect in (False,True):
        for turns in range(4):
            def transform(cell):
                x,y=cell[0]+1,cell[1]+1
                if reflect: x=3-x
                for _ in range(turns): x,y=3-y,x
                return x,y
            variants.append(sorted(connection(transform(a),transform(b)) for a,b in links))
    return hashlib.sha256(json.dumps(min(variants),separators=(',',':')).encode()).hexdigest()


def layout_spec(index):
    if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<24:
        raise ValueError('fixed corpus has layout indices 0..23')
    rng=np.random.default_rng(2026091100+index)
    cells=[(x,y) for x in range(-1,3) for y in range(-1,3)]
    source,junction=(-1,0),(0,0)
    exits=(((0,1),(0,-1)),((1,0),(0,1)),((1,0),(0,-1)),((1,0),(0,1),(0,-1)))[index%4]
    links={connection(source,junction)} | {connection(junction,e) for e in exits}
    parent={c:c for c in cells}
    def root(c):
        while parent[c]!=c: c=parent[c]
        return c
    def merge(a,b): parent[root(a)]=root(b)
    for a,b in links: merge(a,b)
    candidates=[]
    for cell in cells:
        for delta in ((1,0),(0,1)):
            other=(cell[0]+delta[0],cell[1]+delta[1])
            edge=connection(cell,other)
            if other not in parent: continue
            if source in edge and edge!=connection(source,junction): continue
            if junction in edge and edge not in links: continue
            candidates.append(edge)
    rng.shuffle(candidates)
    for a,b in candidates:
        if root(a)!=root(b): links.add(connection(a,b)); merge(a,b)
    for edge in candidates:
        if edge not in links and rng.random()<.12: links.add(edge)
    if len({root(c) for c in cells})!=1: raise ValueError('procedural maze disconnected')
    width=float(rng.uniform(.95,1.25))
    pitch=width+.08
    walls={}
    for cell in cells:
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            neighbor=(cell[0]+dx,cell[1]+dy)
            if connection(cell,neighbor) in links: continue
            midpoint=(2*cell[0]+dx,2*cell[1]+dy)
            key=(midpoint,abs(dx))
            walls[key]={'wall_id':f'wall_{midpoint[0]}_{midpoint[1]}_{abs(dx)}',
                'centre_xyz':[midpoint[0]*pitch/2,midpoint[1]*pitch/2,.3],
                'size_xyz':[.08,pitch+.08,.6] if dx else [pitch+.08,.08,.6],
                'yaw_rad':0.,'material_id':'NEUTRAL_WALL'}
    def region(cell):
        centre=np.array(cell)*pitch
        return [(centre+np.array(offset)*width/2).tolist() for offset in ((-1,-1),(1,-1),(1,1),(-1,1))]
    geometry={'spawn_se2_world':[-pitch,float(rng.uniform(-.07,.07)),float(rng.uniform(-.12,.12))],
        'wall_boxes':[walls[k] for k in sorted(walls)],
        'source_node':{'node_id':'prefix_source','centre_world':[-pitch,0.],'boundary_polygon_world':region(source)},
        'target_node':{'node_id':'junction','centre_world':[0.,0.],'boundary_polygon_world':region(junction)},
        'selected_directed_edge':{'edge_id':'prefix-edge','source_node_id':'prefix_source','target_node_id':'junction',
            'opening_segment_world':[[-pitch/2,-width/2],[-pitch/2,width/2]],'opening_normal_world':[1.,0.],
            'edge_region_polygon_world':region(junction)},'competing_directed_edges':[],
        'teacher_route_polyline_world':[[-pitch,0.],[-pitch/2,0.],[0.,0.]]}
    return {'layout_id':f'counterfactual-maze-development-v1-{index:02d}', 'layout_index':index,
        'data_role':'train' if index<16 else 'validation','procedural_seed':2026091100+index,
        'topology_sha256_dihedral':topology_identity(links),'graph_connections':[[list(a),list(b)] for a,b in sorted(links)],
        'junction_exits':[list(e) for e in exits],'width_m':width,'cell_pitch_m':pitch,'geometry':geometry}


def corpus():
    layouts=[layout_spec(i) for i in range(24)]
    if len({s['topology_sha256_dihedral'] for s in layouts})!=24:
        raise ValueError('duplicate topology in fixed corpus; no runtime resampling')
    return layouts


def branch_spec(layout,action_index):
    if isinstance(action_index,bool) or not isinstance(action_index,int) or not 0<=action_index<len(ACTIONS):
        raise ValueError('unknown branch action')
    name,command=ACTIONS[action_index]
    return {**copy.deepcopy(layout),'scene_id':f'{layout["layout_id"]}-{name}',
        'family':'COUNTERFACTUAL_MAZE_DEVELOPMENT','case_index':layout['layout_index'],'arm':'baseline',
        'action_index':action_index,'action_name':name,'branch_command':list(command)}


def horizon_labels(raw,start_index):
    """Privileged outcome labels only; never fill unobserved future motion."""
    ns=np.rint(raw['timestamp_s']*1e9).astype(np.int64)
    origin=raw['base_pose_world'][start_index]
    rotation=rotation_xyzw(origin[3:])
    yaw0=math.atan2(rotation[1,0],rotation[0,0])
    t0=int(ns[start_index])
    contacts=np.flatnonzero(raw['physics_contact'].astype(bool)&(np.arange(len(ns))>start_index))
    first_contact=None if not len(contacts) else int(ns[contacts[0]]-t0)
    rows=[]
    for horizon in HORIZONS_NS:
        at=np.flatnonzero(ns==t0+horizon)
        valid=bool(len(at))
        motion=None
        if valid:
            pose=raw['base_pose_world'][at[0]]
            r=rotation_xyzw(pose[3:])
            yaw=math.atan2(r[1,0],r[0,0])-yaw0
            delta=rotation.T@(pose[:3]-origin[:3])
            motion=[float(delta[0]),float(delta[1]),math.atan2(math.sin(yaw),math.cos(yaw))]
        observed=int(ns[-1])-t0>=horizon
        rows.append({'horizon_ns':horizon,'motion_valid':valid,'delta_xy_yaw_start_body':motion,
            'contact_valid':bool(observed or (first_contact is not None and first_contact<=horizon)),
            'contact_by_horizon':bool(first_contact is not None and first_contact<=horizon)})
    return rows
