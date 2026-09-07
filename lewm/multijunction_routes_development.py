"""Explicit fresh development route graphs and one shared physical wall scene."""
import copy
import numpy as np

MOTIFS=('left_right','right_left','hairpin','dead_end_return')
WIDTHS=(.9,1.2)
PATHS={
    'left_right':[(0,0),(1,0),(1,1),(2,1)],
    'right_left':[(0,0),(1,0),(1,-1),(2,-1)],
    'hairpin':[(0,0),(1,0),(1,1),(0,1)],
    'dead_end_return':[(0,0),(1,0),(2,0),(1,0),(1,1)],
}
SPURS={
    'left_right':[((1,0),(2,0)),((1,1),(1,2))],
    'right_left':[((1,0),(2,0)),((1,-1),(1,-2))],
    'hairpin':[((1,0),(2,0)),((1,1),(2,1))],
    'dead_end_return':[((1,0),(1,-1))],
}


def connection(a,b):
    return tuple(sorted((tuple(a),tuple(b))))


def route_spec(motif,width):
    if motif not in MOTIFS or width not in WIDTHS:
        raise ValueError('unknown fixed route case')
    index=MOTIFS.index(motif)*2+WIDTHS.index(width)
    path=PATHS[motif]
    links={connection(a,b) for a,b in zip(path,path[1:])} | {connection(a,b) for a,b in SPURS[motif]}
    cells=sorted({cell for edge in links for cell in edge})
    pitch=width+.08
    walls={}
    for cell in cells:
        for direction in ((1,0),(-1,0),(0,1),(0,-1)):
            neighbor=(cell[0]+direction[0],cell[1]+direction[1])
            if connection(cell,neighbor) in links:
                continue
            midpoint=(2*cell[0]+direction[0],2*cell[1]+direction[1])
            key=(midpoint,abs(direction[0]))
            centre=[midpoint[0]*pitch/2,midpoint[1]*pitch/2,.3]
            size=[.08,pitch+.08,.6] if direction[0] else [pitch+.08,.08,.6]
            walls[key]={'wall_id':f'wall_{midpoint[0]}_{midpoint[1]}_{abs(direction[0])}',
                        'centre_xyz':centre,'size_xyz':size,'yaw_rad':0.,'material_id':'NEUTRAL_WALL'}
    walls=[walls[key] for key in sorted(walls)]
    spawn=[0.,.08 if width==.9 else -.08,.12 if width==.9 else -.12]
    def centre(cell): return np.asarray(cell,dtype=float)*pitch
    def region(cell):
        c=centre(cell)
        return [(c+np.array(offset)*width/2).tolist() for offset in ((-1,-1),(1,-1),(1,1),(-1,1))]
    geometries=[]
    for i,(source,target) in enumerate(zip(path,path[1:])):
        s,t=centre(source),centre(target)
        normal=(t-s)/pitch
        tangent=np.array([-normal[1],normal[0]])
        midpoint=(s+t)/2
        opening=np.stack([midpoint-tangent*width/2,midpoint+tangent*width/2])
        route=[s.tolist(),midpoint.tolist(),t.tolist()]
        # Preserve the incoming segment through the cell centre for actual turns.
        # A direct reversal omits that segment to avoid a self-overlapping path.
        if i and path[i-1]!=target:
            previous=(centre(path[i-1])+s)/2
            route.insert(0,previous.tolist())
        geometries.append({'spawn_se2_world':spawn,'wall_boxes':copy.deepcopy(walls),
            'source_node':{'node_id':f'cell_{source[0]}_{source[1]}','centre_world':s.tolist(),'boundary_polygon_world':region(source)},
            'target_node':{'node_id':f'cell_{target[0]}_{target[1]}','centre_world':t.tolist(),'boundary_polygon_world':region(target)},
            'selected_directed_edge':{'edge_id':f'route-edge-{i}',
                'source_node_id':f'cell_{source[0]}_{source[1]}','target_node_id':f'cell_{target[0]}_{target[1]}',
                'opening_segment_world':opening.tolist(),'opening_normal_world':normal.tolist(),
                'edge_region_polygon_world':region(target)},
            'competing_directed_edges':[],'teacher_route_polyline_world':route})
    return {'scene_id':f'go2-multijunction-route-dev-v1-{motif}-width-{int(width*100):03d}',
        'family':'MULTIJUNCTION_ROUTE_DEVELOPMENT','case_index':index,'motif':motif,'width_m':width,
        'procedural_seed':2026091000+index,'arm':'baseline','geometry':copy.deepcopy(geometries[0]),
        'route_geometries':geometries,'route_cells':[list(c) for c in path],
        'graph_connections':[[list(a),list(b)] for a,b in sorted(links)],'cell_pitch_m':pitch}
