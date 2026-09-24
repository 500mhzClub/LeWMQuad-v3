import numpy as np
import pytest

from lewm_genesis.appearance_surface_development import patch,surfaces,triangle_identity,ARMS


def test_appearance_changes_colors_without_changing_surface_triangles():
    meshes=[patch([0,0,0],[1,0,0],[0,1,0],[1,1],arm=a,seed=42) for a in ARMS]
    assert all(triangle_identity(m.vertices,m.faces)==triangle_identity(meshes[0].vertices,meshes[0].faces) for m in meshes)
    assert [len(np.unique(m.visual.vertex_colors,axis=0)) for m in meshes[:2]]==[1,2]
    assert len(np.unique(meshes[2].visual.vertex_colors,axis=0))>20
    np.testing.assert_array_equal(meshes[2].visual.vertex_colors,patch([0,0,0],[1,0,0],[0,1,0],[1,1],arm='distinctive',seed=42).visual.vertex_colors)
    assert not np.array_equal(meshes[2].visual.vertex_colors,patch([0,0,0],[1,0,0],[0,1,0],[1,1],arm='distinctive',seed=43).visual.vertex_colors)


def test_patch_is_exact_planar_tessellation_with_correct_winding():
    mesh=patch([1,2,3],[0,1,0],[0,0,1],[.31,.27],arm='neutral',seed=0)
    np.testing.assert_allclose(mesh.area,.31*.27,atol=1e-12)
    assert np.all(mesh.vertices[:,0]==1)
    np.testing.assert_allclose(mesh.face_normals,np.tile([1,0,0],(len(mesh.faces),1)),atol=1e-12)


def test_six_box_faces_are_coincident_with_physical_box_even_when_rotated():
    box=dict(wall_id='wall',centre_xyz=[1,2,.3],size_xyz=[.08,1.,.6],yaw_rad=.3)
    mesh=surfaces([box],'distinctive')[1][1];c,s=np.cos(.3),np.sin(.3);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    local=(mesh.vertices-box['centre_xyz'])@R;half=np.asarray(box['size_xyz'])/2
    assert np.all(np.abs(local)<=half+1e-12)
    assert np.all(np.min(np.abs(np.abs(local)-half),axis=1)<1e-12)
    np.testing.assert_allclose(mesh.area,2*(.08+ .08*.6+.6),atol=1e-12)


def test_identity_detects_changed_surface_but_not_vertex_reordering(tmp_path):
    mesh=patch([0,0,0],[1,0,0],[0,1,0],[1,1],arm='distinctive',seed=42)
    ident=triangle_identity(mesh.vertices,mesh.faces)
    assert triangle_identity(mesh.vertices,mesh.faces[::-1,::-1])==ident
    changed=mesh.vertices.copy();changed[:,2]+=.005
    assert triangle_identity(changed,mesh.faces)!=ident


@pytest.mark.parametrize('fault',['arm','seed','axes','extent'])
def test_invalid_surface_contract_is_rejected(fault):
    args=dict(origin=[0,0,0],u=[1,0,0],v=[0,1,0],lengths=[1,1],arm='neutral',seed=42)
    if fault=='arm':args['arm']='goal_code'
    if fault=='seed':args['seed']=-1
    if fault=='axes':args['v']=[1,0,0]
    if fault=='extent':args['lengths']=[0,1]
    with pytest.raises(ValueError):patch(**args)
