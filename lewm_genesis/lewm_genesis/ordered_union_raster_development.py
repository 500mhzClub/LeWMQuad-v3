"""Instance-local fixed draw order for exactly two opaque static union surfaces."""
import hashlib
from types import MethodType
import numpy as np

ORDERS = ('floor_first', 'walls_first')


def surface_nodes(scene):
    nodes = list(scene.mesh_nodes)
    if len(nodes) != 2: raise ValueError('exactly two static visual surfaces required')
    by_role = {}; witnesses = {}
    for node in nodes:
        mesh = node.mesh
        if mesh.is_transparent or not mesh.is_visible or len(mesh.primitives) != 1:
            raise ValueError('one visible opaque primitive per surface required')
        pose = np.asarray(scene.get_pose(node))
        if pose.shape != (4,4) or not np.array_equal(pose,np.eye(4)):
            raise ValueError('world-space static surface meshes required')
        primitive = mesh.primitives[0]; p = np.asarray(primitive.positions)
        if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
            raise ValueError('finite native visual vertices required')
        lo,hi = p.min(0),p.max(0)
        if np.array_equal(lo,[-16,-16,0]) and np.array_equal(hi,[16,16,0]): role='floor'
        elif lo[2] == 0 and np.isclose(hi[2],1.4,atol=1e-7,rtol=0) and np.all(np.abs(p[:,:2])<=16): role='walls'
        else: raise ValueError('unexpected union visual geometry')
        if role in by_role: raise ValueError('unique floor and wall roles required')
        by_role[role]=node
        witnesses[role]=dict(vertices=len(p),dtype=str(p.dtype),positions_sha256=hashlib.sha256(p.tobytes()).hexdigest())
    if set(by_role) != {'floor','walls'}: raise ValueError('complete two-surface roles required')
    return by_role,witnesses


def install_order(scene, order):
    """No global/native source patch; positions, materials and pixels are unchanged."""
    if order not in ORDERS: raise ValueError('explicit fixed order required')
    nodes,witnesses=surface_nodes(scene)
    roles=('floor','walls') if order=='floor_first' else ('walls','floor')
    ordered=[nodes[r] for r in roles]
    def fixed_sort(this):
        if set(this.mesh_nodes) != set(ordered): raise ValueError('visual population changed')
        return list(ordered)
    scene.sorted_mesh_nodes=MethodType(fixed_sort,scene)
    scene._meshes_updated=True
    return dict(order=order,roles=list(roles),surfaces=witnesses)


def verify_order(camera, expected):
    context=camera._rasterizer._context
    nodes,witnesses=surface_nodes(context._scene)
    wanted=[nodes[r] for r in expected['roles']]
    if (witnesses!=expected['surfaces'] or list(context.jit.node_list)!=wanted
            or list(context._scene.sorted_mesh_nodes())!=wanted):
        raise ValueError('actual JIT draw order or native surfaces changed')
    return expected
