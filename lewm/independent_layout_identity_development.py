"""Exact bounded graph canonicalization and rigid-grid geometry identity.

Pure construction metadata, never a runtime map or benchmark reader. Refinement
only accelerates exhaustive individualization; a WL hash alone is not identity.
"""
import hashlib
import json


def grid_edges(edges):
    if not isinstance(edges,(list,tuple)) or not edges:raise ValueError('nonempty explicit grid passages required')
    result=[]
    for edge in edges:
        if not isinstance(edge,(list,tuple)) or len(edge)!=2:raise ValueError('two endpoints required')
        nodes=[]
        for p in edge:
            if not isinstance(p,(list,tuple)) or len(p)!=2 or any(type(v) is not int for v in p):
                raise ValueError('integer grid coordinates required')
            nodes.append(tuple(p))
        a,b=nodes
        if sum(abs(x-y) for x,y in zip(a,b))!=1:raise ValueError('unit cardinal passages required')
        result.append(tuple(sorted((a,b))))
    if len(set(result))!=len(result):raise ValueError('duplicate undirected passage')
    result=tuple(sorted(result));nodes=sorted({p for e in result for p in e})
    adjacency={p:set() for p in nodes}
    for a,b in result:adjacency[a].add(b);adjacency[b].add(a)
    reached={nodes[0]}
    while True:
        expanded=reached|{q for p in reached for q in adjacency[p]}
        if expanded==reached:break
        reached=expanded
    if reached!=set(nodes):raise ValueError('connected layout required')
    return result,adjacency


def canonical_graph(adjacency,*,maximum_states=100_000):
    """Exact abstract graph identity, independent of vertex names/embedding.

Adjacency rows refer to0..n-1. Fail closed if the computation budget is exhausted;
never substitute a refinement signature or call an unresolved pair independent.
"""
    n=len(adjacency)
    if not 2<=n<=24 or type(maximum_states) is not int or maximum_states<=0:
        raise ValueError('bounded graph and positive exact-search budget required')
    graph=[]
    for i,row in enumerate(adjacency):
        if (any(type(j) is not int or not 0<=j<n or j==i for j in row)
                or len(set(row))!=len(row) or not 1<=len(row)<=4):raise ValueError('simple bounded-degree adjacency required')
        graph.append(frozenset(row))
    if any(i not in graph[j] for i,row in enumerate(graph) for j in row):raise ValueError('undirected graph required')
    seen={0}
    while True:
        expanded=seen|{j for i in seen for j in graph[i]}
        if expanded==seen:break
        seen=expanded
    if len(seen)!=n:raise ValueError('connected graph required')
    def refine(partition):
        while True:
            refined=[];sets=[set(c) for c in partition]
            for cell in partition:
                groups={}
                for v in cell:groups.setdefault(tuple(len(graph[v]&s) for s in sets),[]).append(v)
                refined.extend(tuple(groups[k]) for k in sorted(groups))
            if len(refined)==len(partition):return tuple(refined)
            partition=tuple(refined)
    states=0;best=None;best_order=None
    def visit(partition):
        nonlocal states,best,best_order
        states+=1
        if states>maximum_states:raise ValueError('exact topology search budget exhausted; identity unresolved')
        partition=refine(partition)
        tied=[(len(c),i) for i,c in enumerate(partition) if len(c)>1]
        if not tied:
            order=tuple(c[0] for c in partition)
            code=''.join('1' if order[j] in graph[order[i]] else '0' for i in range(n) for j in range(i+1,n))
            if best is None or code<best:best,best_order=code,order
            return
        _,at=min(tied);cell=partition[at]
        for v in cell:visit(partition[:at]+((v,),tuple(w for w in cell if w!=v))+partition[at+1:])
    visit(tuple(tuple(i for i in range(n) if len(graph[i])==d) for d in sorted({len(g) for g in graph})))
    code=f'undirected-v1:{n}:{best}'
    return dict(code=code,sha256=hashlib.sha256(code.encode()).hexdigest(),canonical_vertex_order=list(best_order),search_states=states)


def topology_identity(edges):
    _,adjacency=grid_edges(edges);nodes=sorted(adjacency);indices={p:i for i,p in enumerate(nodes)}
    return canonical_graph([sorted(indices[q] for q in adjacency[p]) for p in nodes])


def metric_identity(edges,*,pitch_m=1.2,wall_thickness_m=.08,wall_height_m=1.4):
    """Translation/reflection/right-angle rotation invariant square-grid geometry."""
    edges,_=grid_edges(edges)
    for v in (pitch_m,wall_thickness_m,wall_height_m):
        if type(v) not in (int,float) or not 0<float(v)<100:raise ValueError('finite positive metric geometry required')
    variants=[]
    for swap in (False,True):
        for sx in (-1,1):
            for sy in (-1,1):
                transform=lambda p:(sx*p[1],sy*p[0]) if swap else (sx*p[0],sy*p[1])
                pairs=[(transform(a),transform(b)) for a,b in edges]
                lo=tuple(min(p[i] for e in pairs for p in e) for i in range(2))
                pairs=tuple(sorted(tuple(sorted(tuple(p[i]-lo[i] for i in range(2)) for p in e)) for e in pairs))
                variants.append(pairs)
    value=dict(edges=min(variants),pitch_m=float(pitch_m).hex(),wall_thickness_m=float(wall_thickness_m).hex(),
        wall_height_m=float(wall_height_m).hex())
    code=json.dumps(value,sort_keys=True,separators=(',',':'))
    return dict(code=code,sha256=hashlib.sha256(code.encode()).hexdigest())


def verify_role_separation(layouts):
    """Compute identities from supplied edges; never trust claimed layout IDs."""
    by_topology={};by_metric={};rows=[]
    for layout in layouts:
        role=layout['role']
        if role not in ('train','selection','development_eval'):raise ValueError('explicit development role required')
        top=topology_identity(layout['edges']);metric=metric_identity(layout['edges'],pitch_m=layout['pitch_m'])
        for groups,code in ((by_topology,top['code']),(by_metric,metric['code'])):
            if code in groups and groups[code]!=role:raise ValueError('equivalent layouts cannot cross roles')
            groups[code]=role
        rows.append(dict(role=role,topology_sha256=top['sha256'],metric_sha256=metric['sha256']))
    return dict(layouts=rows,topology_groups=len(by_topology),metric_groups=len(by_metric),cross_role_duplicates=0,
        exact_graph_isomorphism_checked=True,native_geometry_verified=False,wall_roster_verified=False,final_evaluation=False)
