"""Boundary-only visuals for equal-height axis-aligned maze wall unions.

Physical boxes remain unchanged. Integer-micrometre construction avoids tiny
floating slivers; unsupported geometry is rejected instead of approximated.
There are no duplicate-area coplanar faces or internal faces. Native rendering
and camera equivalence still require a separate bench before collection use.
"""
import hashlib
import json
import numpy as np
import trimesh
from lewm_genesis.appearance_surface_development import ARMS,patch


def wall_union_boundary(boxes):
    if not isinstance(boxes,(list,tuple)) or not 1<=len(boxes)<=128:
        raise ValueError('bounded nonempty explicit wall box roster required')
    bounds={}
    for box in boxes:
        name=box['wall_id'];centre=np.asarray(box['centre_xyz'],float);size=np.asarray(box['size_xyz'],float)
        if (not isinstance(name,str) or not name or name in bounds or centre.shape!=(3,) or size.shape!=(3,)
                or not np.isfinite(centre).all() or not np.isfinite(size).all() or (size<=0).any()
                or box['yaw_rad']!=0):
            raise ValueError('unique finite positive axis-aligned wall boxes required')
        value=np.stack([centre-size/2,centre+size/2])
        if np.abs(value).max()>16:raise ValueError('walls must fit the bounded16m visual domain')
        rounded=np.rint(value*1e6)
        if not np.allclose(value,rounded/1e6,rtol=0,atol=1e-9):
            raise ValueError('exact micrometre-grid geometry required, no rounded substitute')
        b=rounded.astype(np.int64)
        if (b[1]<=b[0]).any():raise ValueError('positive resolved wall dimensions required')
        bounds[name]=b
    z_ranges={tuple(b[:,2]) for b in bounds.values()}
    if len(z_ranges)!=1:raise ValueError('one common wall base and top required')
    z0,z1=next(iter(z_ranges))
    if z0!=0:raise ValueError('maze wall union must meet the z0 physical floor')
    x=sorted({int(v) for b in bounds.values() for v in b[:,0]})
    y=sorted({int(v) for b in bounds.values() for v in b[:,1]})
    if (len(x)-1)*(len(y)-1)>65536:raise ValueError('bounded coordinate-compressed footprint required')
    names=sorted(bounds);owner=np.full((len(x)-1,len(y)-1),-1,dtype=np.int64)
    for i,name in enumerate(names):
        b=bounds[name];lo=[x.index(int(b[0,0])),y.index(int(b[0,1]))];hi=[x.index(int(b[1,0])),y.index(int(b[1,1]))]
        cell=owner[lo[0]:hi[0],lo[1]:hi[1]];cell[cell<0]=i
    faces=[]
    def add(axis,plane,u0,u1,v0,v1,sign,own):
        faces.append(dict(axis=axis,plane_um=int(plane),u0_um=int(u0),u1_um=int(u1),
            v0_um=int(v0),v1_um=int(v1),normal_sign=sign,owner_wall_id=names[int(own)]))
    for i,j in zip(*np.nonzero(owner>=0),strict=True):
        own=owner[i,j]
        if i==0 or owner[i-1,j]<0:add(0,x[i],y[j],y[j+1],z0,z1,-1,own)
        if i==len(x)-2 or owner[i+1,j]<0:add(0,x[i+1],y[j],y[j+1],z0,z1,1,own)
        # Cyclic tangent axes for y-normal faces are(z,x).
        if j==0 or owner[i,j-1]<0:add(1,y[j],z0,z1,x[i],x[i+1],-1,own)
        if j==len(y)-2 or owner[i,j+1]<0:add(1,y[j+1],z0,z1,x[i],x[i+1],1,own)
        add(2,z0,x[i],x[i+1],y[j],y[j+1],-1,own)
        add(2,z1,x[i],x[i+1],y[j],y[j+1],1,own)
    faces.sort(key=lambda f:tuple(f[k] for k in ('axis','plane_um','normal_sign','u0_um','u1_um','v0_um','v1_um','owner_wall_id')))
    area=sum((f['u1_um']-f['u0_um'])*(f['v1_um']-f['v0_um']) for f in faces)/1e12
    footprint=sum((x[i+1]-x[i])*(y[j+1]-y[j]) for i,j in zip(*np.nonzero(owner>=0),strict=True))/1e12
    return dict(faces=faces,union_surface_area_m2=float(area),union_footprint_area_m2=float(footprint),
        input_wall_ids=names,physical_box_geometry_changed=False,coordinate_grid_m=1e-6,
        native_visibility_qualified=False)


def rectangle_patch(face,*,arm,seed):
    axis=face['axis'];j,k=(axis+1)%3,(axis+2)%3;sign=face['normal_sign']
    origin=np.zeros(3);origin[axis]=face['plane_um']/1e6;origin[j]=face['u0_um']/1e6
    origin[k]=face['v0_um' if sign==1 else 'v1_um']/1e6
    u=np.eye(3)[j];v=sign*np.eye(3)[k]
    lengths=np.array([face['u1_um']-face['u0_um'],face['v1_um']-face['v0_um']])/1e6
    return patch(origin,u,v,lengths,arm=arm,seed=seed,neutral=89)


def independently_seeded_union_surfaces(boxes,arm,seed):
    if arm not in ARMS or type(seed) is not int or seed<0:raise ValueError('explicit independent appearance required')
    boundary=wall_union_boundary(boxes);meshes=[]
    for face in boundary['faces']:
        identity=json.dumps(dict(appearance_seed=seed,face=face),sort_keys=True,separators=(',',':')).encode()
        face_seed=int.from_bytes(hashlib.sha256(identity).digest()[:8],'big')
        meshes.append(rectangle_patch(face,arm=arm,seed=face_seed))
    return [('ground_visual',patch([-16,-16,0],[1,0,0],[0,1,0],[32,32],arm=arm,seed=seed)),
        ('wall_union_visual',trimesh.util.concatenate(meshes))]
