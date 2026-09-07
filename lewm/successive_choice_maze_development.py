"""Fixed fresh topology panel and non-privileged successive command loop."""
import copy

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.counterfactual_maze_development import corpus as training_corpus,topology_identity
from lewm.multijunction_routes_development import connection
from lewm.online_choice_maze_pilot_development import corpus as pilot_corpus,INTENTS
from lewm.online_temporal_choice_development import METHODS


def layout_spec(index):
    if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<8:
        raise ValueError('fixed successive panel has eight layouts')
    seed=2026091800+index; rng=np.random.default_rng(seed)
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
    if len({root(c) for c in cells})!=1: raise ValueError('disconnected fixed topology')
    width=float(rng.uniform(.95,1.25)); pitch=width+.08; walls={}
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
    return {'layout_id':f'successive-choice-maze-development-v1-{index:02d}','layout_index':index,
        'data_role':'successive_online_development','procedural_seed':seed,'topology_sha256_dihedral':topology_identity(links),
        'graph_connections':[[list(a),list(b)] for a,b in sorted(links)],'junction_exits':[list(e) for e in exits],
        'width_m':width,'cell_pitch_m':pitch,'geometry':geometry}


def corpus():
    layouts=[layout_spec(i) for i in range(8)]; identities={r['topology_sha256_dihedral'] for r in layouts}
    prior={r['topology_sha256_dihedral'] for r in training_corpus()+pilot_corpus()}
    if len(identities)!=8 or identities & prior: raise ValueError('fixed topology overlap; no runtime resampling')
    return layouts


def trials():
    return [{**copy.deepcopy(layout),'scene_id':f'{layout["layout_id"]}-{intent}-{method}',
        'family':'SUCCESSIVE_DIRECTIONAL_CHOICE_DEVELOPMENT','case_index':layout['layout_index'],'arm':'baseline',
        'method':method,'intent_name':intent,'intent_xy_body_start_m':list(direction)}
        for layout in corpus() for intent,direction in INTENTS for method in METHODS]


def execute_control(policy,*,observe,step,clock,record_selection,record_command,record_fault):
    """Eight choices, five ticks each, then five zero-command release ticks.

    Callbacks expose only causal packets/clock and commanded motion. The caller
    initializes the direction after its teacher prefix. Native physical stops
    propagate immediately; sensor/model contract errors cancel all further
    learned motion and request a five-tick zero release. Step must disable
    observation ingestion for release so a failed adapter cannot retain motion.
    Every command is recorded BEFORE dispatch, including interrupted ticks.
    """
    fault=None; completed_ticks=0; selections=0
    try:
        for index in range(8):
            choice=policy.select(now_ns=clock()); record_selection(choice); selections+=1
            if choice['decision_index']!=index or len(choice['requested_command_tape'])!=5:
                raise SensorContractError('invalid five-tick sequential choice')
            for command in choice['requested_command_tape']:
                record_command(command,'control',index)
                step(command,False); completed_ticks+=1; observe()
    except SensorContractError as error:
        fault=str(error); record_fault(fault)
    for _ in range(5):
        record_command([0.,0.,0.],'fault_release' if fault else 'release',None)
        step([0.,0.,0.],True)
    return {'sensor_fault':fault,'completed_control_ticks':completed_ticks,'selections':selections,'release_ticks':5}
