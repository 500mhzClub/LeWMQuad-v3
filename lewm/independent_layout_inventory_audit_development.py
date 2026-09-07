"""Independent pairwise adjacency matching and explicit inventory accounting."""
from collections import Counter
from itertools import combinations
from lewm.independent_layout_identity_development import grid_edges
from lewm.independent_layout_inventory_development import CONTEXTS,HISTORIES,SUPPORTS


def pairwise_isomorphic(left,right,*,maximum_states=100_000):
    """Backtracking adjacency bijection; no canonicalization/refinement hashes."""
    _,a=grid_edges(left);_,b=grid_edges(right)
    if type(maximum_states) is not int or maximum_states<=0:raise ValueError('positive comparison budget required')
    if len(a)!=len(b) or Counter(map(len,a.values()))!=Counter(map(len,b.values())):
        return dict(isomorphic=False,search_states=0,mapping=None)
    options={u:tuple(v for v in sorted(b) if len(a[u])==len(b[v])) for u in a}
    mapping={};used=set();states=0
    def search():
        nonlocal states
        states+=1
        if states>maximum_states:raise ValueError('pairwise isomorphism unresolved at search budget')
        if len(mapping)==len(a):return True
        u=min((u for u in a if u not in mapping),key=lambda u:(-sum(v in mapping for v in a[u]),-len(a[u]),u))
        for v in options[u]:
            if v in used or any((w in a[u])!=(mapped in b[v]) for w,mapped in mapping.items()):continue
            mapping[u]=v;used.add(v)
            if search():return True
            del mapping[u];used.remove(v)
        return False
    matched=search()
    return dict(isomorphic=matched,search_states=states,mapping=[[list(k),list(v)] for k,v in sorted(mapping.items())] if matched else None)


def audit_inventory(inventory):
    layouts=inventory['layouts'];episodes=inventory['episodes'];comparisons=[]
    if len(layouts)!=12 or len(episodes)!=1440:raise ValueError('complete fixed inventory required')
    ids={x['layout_id']:x for x in layouts}
    if len(ids)!=len(layouts):raise ValueError('unique layout identities required')
    if Counter(l['role'] for l in layouts)!=dict(train=6,selection=3,development_eval=3):raise ValueError('prospective6/3/3 roles required')
    for a,b in combinations(layouts,2):
        result=pairwise_isomorphic(a['edges'],b['edges'])
        if result['isomorphic']:raise ValueError('inventory includes an abstract-topology duplicate')
        comparisons.append(dict(left=a['layout_id'],right=b['layout_id'],cross_role=a['role']!=b['role'],**result))
    groups={};episode_ids=set()
    for e in episodes:
        if e['episode_id'] in episode_ids:raise ValueError('unique episode IDs required')
        episode_ids.add(e['episode_id']);layout=ids[e['layout_id']]
        if e['role']!=layout['role'] or e['topology_group_id']!=layout['topology_group_id']:raise ValueError('episode role/group differs from entire layout')
        if e['context_kind'] not in CONTEXTS or e['history_kind'] not in HISTORIES or e['support'] not in SUPPORTS:
            raise ValueError('declared factorial context required')
        groups.setdefault((e['layout_id'],e['context_kind'],e['history_kind'],e['support']),[]).append(e)
    if len(groups)!=240:raise ValueError('all240 six-action context groups required')
    for group in groups.values():
        if len(group)!=6 or {e['action_index'] for e in group}!=set(range(6)):raise ValueError('all six action cells required')
        fields=('physics_seed','appearance_seed','spawn_se2_world','spawn_z_m','warmup_ticks','warmup_command','departure_ns','friction_mu')
        if any(e[k]!=group[0][k] for e in group for k in fields):raise ValueError('planned sibling construction differs')
    return dict(status='PROSPECTIVE_INVENTORY_STRUCTURAL_AUDIT_PASS',layout_pairs_checked=len(comparisons),comparisons=comparisons,
        independent_abstract_topologies=12,planned_six_action_groups=240,planned_episodes=1440,
        role_counts=dict(Counter(l['role'] for l in layouts)),episode_role_counts=dict(Counter(e['role'] for e in episodes)),
        actual_prefix_matching_verified=False,native_setup_verified=False,positive_contact_coverage_verified=False,
        physics_collected=False,final_evaluation=False,navigation_qualified=False,goal_achieved=False)
