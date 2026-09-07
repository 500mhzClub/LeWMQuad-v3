"""Nonsemantic vertex-color appearance on explicit, unchanged surface geometry."""
import hashlib

import numpy as np
import trimesh

ARMS=('neutral','repeated','distinctive')
CELL_M=.125
APPEARANCE_SEED=271828


def patch(origin,u,v,lengths,*,arm,seed,neutral=128):
    origin,u,v=[np.asarray(x,float) for x in (origin,u,v)];lengths=np.asarray(lengths,float)
    if (arm not in ARMS or type(seed) is not int or seed<0 or any(x.shape!=(3,) for x in (origin,u,v))
            or lengths.shape!=(2,) or not np.isfinite([origin,u,v]).all() or not np.isfinite(lengths).all()
            or np.any(lengths<=0) or not np.allclose([u@u,v@v,u@v],[1,1,0],atol=1e-12,rtol=0)):
        raise ValueError('explicit finite orthonormal surface and independent appearance seed required')
    n,m=np.ceil(lengths/CELL_M).astype(int)
    x,y=np.meshgrid(np.linspace(0,lengths[0],n+1),np.linspace(0,lengths[1],m+1))
    p=origin+x[...,None]*u+y[...,None]*v
    quads=np.stack((p[:-1,:-1],p[:-1,1:],p[1:,1:],p[1:,:-1]),axis=2).reshape(-1,4,3)
    vertices=quads.reshape(-1,3);base=np.arange(len(quads))*4
    faces=np.stack((base[:,None]+[0,1,2],base[:,None]+[0,2,3]),axis=1).reshape(-1,3)
    if arm=='neutral':gray=np.full(len(quads),neutral,np.uint8)
    elif arm=='repeated':gray=np.where(np.indices((m,n)).sum(0).ravel()%2,205,50).astype(np.uint8)
    else:gray=np.random.default_rng(seed).integers(40,231,len(quads),dtype=np.uint8)
    color=np.column_stack((np.repeat(gray,4)[:,None]*np.ones((1,3),np.uint8),np.full(len(vertices),255,np.uint8)))
    return trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=color,process=False)


def surfaces(boxes,arm):
    result=[('ground_visual',patch([-16,-16,0],[1,0,0],[0,1,0],[32,32],arm=arm,seed=APPEARANCE_SEED))]
    for i,box in enumerate(boxes):
        centre=np.asarray(box['centre_xyz']);half=np.asarray(box['size_xyz'])/2
        angle=box['yaw_rad'];c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        meshes=[]
        for axis in range(3):
            for sign in (-1,1):
                j,k=(axis+1)%3,(axis+2)%3;u=R[:,j];v=sign*R[:,k]
                origin=centre+sign*half[axis]*R[:,axis]-half[j]*u-half[k]*v
                meshes.append(patch(origin,u,v,2*half[[j,k]],arm=arm,
                    seed=APPEARANCE_SEED+1+i*6+axis*2+(sign==1),neutral=89))
        result.append((box['wall_id']+'_visual',trimesh.util.concatenate(meshes)))
    return result


def triangle_identity(vertices,faces):
    """Winding-independent triangle multiset; fixed1um numerical readback grid."""
    vertices=np.asarray(vertices,float);faces=np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or faces.ndim!=2 or faces.shape[1:]!=(3,)
            or not np.isfinite(vertices).all() or faces.dtype.kind not in 'iu'
            or faces.min()<0 or faces.max()>=len(vertices)):
        raise ValueError('finite indexed triangle mesh required')
    triangles=np.rint(vertices[faces]*1e6).astype(np.int64)
    # Canonicalize vertex order within each triangle, then triangle order.
    order=np.lexsort((triangles[:,:,2],triangles[:,:,1],triangles[:,:,0]),axis=1)
    triangles=np.take_along_axis(triangles,order[:,:,None],axis=1).reshape(-1,9)
    triangles=triangles[np.lexsort(tuple(triangles[:,i] for i in range(8,-1,-1)))]
    area=np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)/2
    return dict(triangles=len(faces),triangle_multiset_sha256=hashlib.sha256(triangles.tobytes()).hexdigest(),
        minimum_xyz=vertices.min(0).tolist(),maximum_xyz=vertices.max(0).tolist(),area_m2=float(area.sum()))
