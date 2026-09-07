from types import SimpleNamespace as NS
import numpy as np
import pytest
from lewm_genesis.ordered_union_raster_development import install_order,surface_nodes,verify_order

class Node:
    def __init__(self,vertices):
        self.mesh=NS(is_transparent=False,is_visible=True,primitives=[NS(positions=np.asarray(vertices,np.float32))])

class Scene:
    def __init__(self):
        self.floor=Node([[-16,-16,0],[16,16,0]])
        self.walls=Node([[-2,-2,0],[3,3,1.4]])
        self.mesh_nodes={self.floor,self.walls}
        self.pose=np.eye(4)
    def get_pose(self,node): return self.pose

@pytest.mark.parametrize('order,roles',[('floor_first',['floor','walls']),('walls_first',['walls','floor'])])
def test_explicit_order_and_actual_jit_readback(order,roles):
    s=Scene();w=install_order(s,order);assert w['roles']==roles and s._meshes_updated
    context=NS(_scene=s,jit=NS(node_list=s.sorted_mesh_nodes()))
    camera=NS(_rasterizer=NS(_context=context))
    assert verify_order(camera,w)==w
    context.jit.node_list.reverse()
    with pytest.raises(ValueError):verify_order(camera,w)

def test_identity_is_independent_of_instance_and_set_order():
    a,b=Scene(),Scene()
    assert install_order(a,'floor_first')==install_order(b,'floor_first')
    assert 'sorted_mesh_nodes' not in Scene.__dict__

def test_visual_population_change_fails():
    s=Scene();install_order(s,'floor_first');s.mesh_nodes.remove(s.floor)
    with pytest.raises(ValueError):s.sorted_mesh_nodes()

@pytest.mark.parametrize('change',['transparent','hidden','pose','vertices','extra','duplicate','empty','height'])
def test_reject_unsupported_scene(change):
    s=Scene()
    if change=='transparent':s.floor.mesh.is_transparent=True
    elif change=='hidden':s.floor.mesh.is_visible=False
    elif change=='pose':s.pose[0,3]=.1
    elif change=='vertices':s.floor.mesh.primitives[0].positions[0,0]=np.nan
    elif change=='extra':s.mesh_nodes.add(Node([[0,0,0],[1,1,1.4]]))
    elif change=='duplicate':s.walls.mesh.primitives[0].positions=s.floor.mesh.primitives[0].positions.copy()
    elif change=='empty':s.floor.mesh.primitives=[]
    elif change=='height':s.walls.mesh.primitives[0].positions[1,2]=1.5
    with pytest.raises(ValueError):surface_nodes(s)

def test_unknown_order_rejected_before_mutation():
    s=Scene()
    with pytest.raises(ValueError):install_order(s,'whatever')
    assert not hasattr(s,'sorted_mesh_nodes')
