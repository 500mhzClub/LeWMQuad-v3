"""Exact grounded axis-aligned box-union surfaces, including unequal heights.

Coordinate-compressed3D cells retain original micrometre-grid geometry. Only
occupied/unoccupied interfaces become faces; overlapping boxes have no internal
surfaces. Physical boxes are unchanged. Native visibility requires its own bench.
"""
import hashlib
import json
import math
import numpy as np
import trimesh
from lewm_genesis.union_wall_surface_development import wall_union_boundary as validate_single, rectangle_patch
from lewm_genesis.appearance_surface_development import ARMS,patch


def wall_union_boundary(boxes):
    if not isinstance(boxes,(list,tuple)) or not 1<=len(boxes)<=128:
        raise ValueError('bounded explicit wall roster required')
    bounds={}
    for box in boxes:
        # Reuse existing exact grid, grounded-base, finite/axis-aligned checks.
        validate_single([box]);name=box['wall_id']
        if name in bounds:raise ValueError('unique wall identities required')
        center=np.asarray(box['centre_xyz'],float);half=np.asarray(box['size_xyz'],float)/2
        bounds[name]=np.rint(np.stack([center-half,center+half])*1e6).astype(np.int64)
    coordinates=[sorted({int(v) for b in bounds.values() for v in b[:,a]}) for a in range(3)]
    shape=tuple(len(c)-1 for c in coordinates)
    if math.prod(shape)>65536:raise ValueError('bounded65536cell coordinate-compressed union required')
    names=sorted(bounds);owner=np.full(shape,-1,dtype=np.int64)
    for i,name in enumerate(names):
        b=bounds[name]
        lo=[coordinates[a].index(int(b[0,a])) for a in range(3)]
        hi=[coordinates[a].index(int(b[1,a])) for a in range(3)]
        cells=owner[tuple(slice(l,h) for l,h in zip(lo,hi,strict=True))];cells[cells<0]=i
    faces=[];volume_um3=0
    for index in zip(*np.nonzero(owner>=0),strict=True):
        volume_um3+=math.prod(int(coordinates[a][index[a]+1]-coordinates[a][index[a]]) for a in range(3))
        for axis in range(3):
            j,k=(axis+1)%3,(axis+2)%3
            for sign in (-1,1):
                adjacent=list(index);adjacent[axis]+=sign
                if 0<=adjacent[axis]<shape[axis] and owner[tuple(adjacent)]>=0:continue
                faces.append(dict(axis=axis,plane_um=coordinates[axis][index[axis]+int(sign>0)],
                    u0_um=coordinates[j][index[j]],u1_um=coordinates[j][index[j]+1],
                    v0_um=coordinates[k][index[k]],v1_um=coordinates[k][index[k]+1],
                    normal_sign=sign,owner_wall_id=names[int(owner[index])]))
    fields=('axis','plane_um','normal_sign','u0_um','u1_um','v0_um','v1_um','owner_wall_id')
    faces.sort(key=lambda f:tuple(f[k] for k in fields))
    area=sum((f['u1_um']-f['u0_um'])*(f['v1_um']-f['v0_um']) for f in faces)/1e12
    footprint=sum((coordinates[0][i+1]-coordinates[0][i])*(coordinates[1][j+1]-coordinates[1][j])
        for i,j in zip(*np.nonzero((owner>=0).any(2)),strict=True))/1e12
    return dict(faces=faces,union_surface_area_m2=float(area),union_footprint_area_m2=float(footprint),
        union_volume_m3=volume_um3/1e18,compressed_cells=math.prod(shape),input_wall_ids=names,
        physical_box_geometry_changed=False,coordinate_grid_m=1e-6,native_visibility_qualified=False)


def independently_seeded_union_surfaces(boxes,arm,seed):
    if arm not in ARMS or type(seed) is not int or seed<0:raise ValueError('explicit independent appearance required')
    boundary=wall_union_boundary(boxes);meshes=[]
    for face in boundary['faces']:
        identity=json.dumps(dict(appearance_seed=seed,face=face),sort_keys=True,separators=(',',':')).encode()
        face_seed=int.from_bytes(hashlib.sha256(identity).digest()[:8],'big')
        meshes.append(rectangle_patch(face,arm=arm,seed=face_seed))
    return [('ground_visual',patch([-16,-16,0],[1,0,0],[0,1,0],[32,32],arm=arm,seed=seed)),
        ('wall_union_visual',trimesh.util.concatenate(meshes))]
