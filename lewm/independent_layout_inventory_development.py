"""Prospective12-layout/1440-episode development inventory; no execution or fitting."""
import math
import random
import json
from collections import Counter
from lewm.independent_layout_identity_development import grid_edges,topology_identity,metric_identity,verify_role_separation
from lewm.coupled_pulse_rollout_development import COMMANDS

CONSTRUCTION_SEED=2026090803
ROLE_SEED=2026090804
PITCH=1.2
CONTEXTS=('open_passage','corner_turn','junction','dead_end_approach','near_wall')
HISTORIES=('quiet','recent_forward')
SUPPORTS=('nominal','lower_friction')
ROLE_ORDER=('train','selection','development_eval')


def candidate(rng,cycles):
    cells=[(x,y) for x in range(-1,3) for y in range(-1,3)]
    edges=[(p,q) for p in cells for q in cells if p<q and sum(abs(a-b) for a,b in zip(p,q))==1]
    rng.shuffle(edges);parent={p:p for p in cells};tree=[];extra=[]
    def root(p):
        while parent[p]!=p:p=parent[p]
        return p
    for a,b in edges:
        if root(a)!=root(b):parent[root(a)]=root(b);tree.append((a,b))
        else:extra.append((a,b))
    rng.shuffle(extra);return grid_edges(tree+extra[:cycles])[0]


def cell_types(edges):
    _,adj=grid_edges(edges);types={k:[] for k in ('straight','corner','junction','dead_end')}
    for p,neighbours in sorted(adj.items()):
        if len(neighbours)==1:types['dead_end'].append(p)
        elif len(neighbours)>=3:types['junction'].append(p)
        else:
            a,b=sorted(neighbours)
            types['straight' if (a[0]+b[0],a[1]+b[1])==(2*p[0],2*p[1]) else 'corner'].append(p)
    return types


def wall_boxes(edges):
    edges,adj=grid_edges(edges);passages={frozenset(e) for e in edges};walls={}
    for x,y in sorted(adj):
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            if frozenset(((x,y),(x+dx,y+dy))) in passages:continue
            mid=(2*x+dx,2*y+dy);key=(mid,abs(dx))
            walls[key]=dict(wall_id=f'inventory_wall_{mid[0]}_{mid[1]}_{abs(dx)}',
                centre_xyz=[mid[0]*PITCH/2,mid[1]*PITCH/2,.7],
                size_xyz=[.08,PITCH+.08,1.4] if dx else [PITCH+.08,.08,1.4],yaw_rad=0.,material_id='NEUTRAL_WALL')
    return [walls[k] for k in sorted(walls)]


def layout_contexts(layout):
    _,adj=grid_edges(layout['edges']);types=cell_types(layout['edges']);contexts=[]
    rng=random.Random(2026093000+layout['layout_index'])
    chosen={k:rng.choice(v) for k,v in types.items()}
    for i,kind in enumerate(CONTEXTS):
        key={'open_passage':'straight','corner_turn':'corner','junction':'junction',
             'dead_end_approach':'dead_end','near_wall':'dead_end'}[kind]
        p=chosen[key];q=rng.choice(sorted(adj[p]));direction=(q[0]-p[0],q[1]-p[1])
        if key=='dead_end':direction=(-direction[0],-direction[1])
        offset=.16 if kind=='near_wall' else 0.
        contexts.append(dict(kind=kind,cell=list(p),forward_grid=list(direction),
            spawn_se2_world=[p[0]*PITCH+direction[0]*offset,p[1]*PITCH+direction[1]*offset,math.atan2(direction[1],direction[0])],
            spawn_z_m=.375,physics_seed=2026091000+layout['layout_index']*10+i,
            appearance_seed=2026092000+layout['layout_index']*10+i,initial_clearance_requires_native_setup_check=True))
    return contexts


def build_inventory():
    rng=random.Random(CONSTRUCTION_SEED);buckets={1:[],2:[],3:[]};codes=set();attempted=0;rejected=Counter()
    while any(len(v)<4 for v in buckets.values()):
        cycles=next(k for k,v in buckets.items() if len(v)<4);attempted+=1
        if attempted>10000:raise ValueError('bounded structural inventory construction exhausted')
        edges=candidate(rng,cycles)
        if any(not v for v in cell_types(edges).values()):rejected['missing_context_cell_type']+=1;continue
        topology=topology_identity(edges)
        if topology['code'] in codes:rejected['isomorphic_duplicate']+=1;continue
        codes.add(topology['code']);buckets[cycles].append((edges,topology))
    layouts=[];roles=random.Random(ROLE_SEED)
    for cycles,values in buckets.items():
        roles.shuffle(values)
        for role,(edges,topology) in zip(('train','train','selection','development_eval'),values,strict=True):
            metric=metric_identity(edges)
            layouts.append(dict(role=role,edges=[[list(a),list(b)] for a,b in edges],pitch_m=PITCH,
                layout_id='connected-inventory-v1-'+metric['sha256'],topology_group_id='topology-v1-'+topology['sha256'],
                topology_canonical_code=topology['code'],metric_canonical_code=metric['code'],cycle_rank=cycles,
                cells=[list(p) for p in sorted({p for e in edges for p in e})],wall_boxes=wall_boxes(edges)))
    layouts.sort(key=lambda x:(ROLE_ORDER.index(x['role']),x['cycle_rank'],x['layout_id']))
    for i,row in enumerate(layouts):row['layout_index']=i;row['contexts']=layout_contexts(row)
    episodes=[]
    for layout in layouts:
        for context in layout['contexts']:
            for history in HISTORIES:
                for support in SUPPORTS:
                    for action in range(6):
                        episodes.append(dict(episode_id=f"l{layout['layout_index']:02d}_{context['kind']}_{history}_{support}_a{action}",
                            layout_id=layout['layout_id'],topology_group_id=layout['topology_group_id'],role=layout['role'],
                            context_kind=context['kind'],history_kind=history,support=support,
                            physics_seed=context['physics_seed'],appearance_seed=context['appearance_seed'],
                            spawn_se2_world=list(context['spawn_se2_world']),spawn_z_m=context['spawn_z_m'],
                            friction_mu=1. if support=='nominal' else .15,
                            warmup_ticks=8,warmup_command=[0.,0.,0.] if history=='quiet' else [.12,0.,0.],
                            action_index=action,command=list(COMMANDS[action//2]),pulse_ticks=(2,5)[action%2],brake_ticks=20,
                            departure_tick=8,departure_ns=2_300_000_000))
    separation=verify_role_separation(layouts)
    return dict(schema='prospective_connected_layout_inventory_development.v1',construction_seed=CONSTRUCTION_SEED,role_seed=ROLE_SEED,
        layouts=layouts,episodes=episodes,role_separation=separation,structural_candidates=attempted,structural_rejections=dict(rejected),
        role_counts=dict(Counter(x['role'] for x in layouts)),episode_role_counts=dict(Counter(x['role'] for x in episodes)),
        scientific_unit='complete_layout_topology_group',physics_collected=False,model_trained=False,
        hardware_validated=False,final_evaluation=False,navigation_qualified=False,goal_achieved=False)


def validate_inventory(inventory):
    """Require the exact prospective construction, not merely self-consistent IDs."""
    expected=build_inventory()
    if json.dumps(inventory,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('exact prospective layout/context/action/role inventory required')
    return inventory['role_separation']
