"""Analytic volume/boundary and exact real-pilot visual construction checks."""
from copy import deepcopy
from pathlib import Path
import ast
import numpy as np
import pytest
from lewm_genesis.variable_height_union_surface_development import wall_union_boundary,independently_seeded_union_surfaces
from lewm_genesis.union_wall_surface_development import wall_union_boundary as original
from lewm_genesis.appearance_surface_development import triangle_identity
from lewm.geometry_progress_near_field_development import GEOMETRIES,TRIALS,specification
from lewm.tests.test_union_wall_surface_development import no_overlapping_coplanar_faces


def box(name,x,h):return dict(wall_id=name,centre_xyz=[x,.5,h/2],size_xyz=[1.,1.,h],yaw_rad=0.)


def assert_boundary(boxes,r):
    no_overlapping_coplanar_faces(r['faces'])
    centers=np.array([b['centre_xyz'] for b in boxes]);half=np.array([b['size_xyz'] for b in boxes])/2
    for f in r['faces']:
        a=f['axis'];j,k=(a+1)%3,(a+2)%3;p=np.zeros(3)
        p[a]=f['plane_um']/1e6;p[j]=(f['u0_um']+f['u1_um'])/2e6;p[k]=(f['v0_um']+f['v1_um'])/2e6
        n=np.eye(3)[a]*f['normal_sign']
        assert np.any(np.all(np.abs(p-1e-7*n-centers)<half,axis=1))
        assert not np.any(np.all(np.abs(p+1e-7*n-centers)<half,axis=1))
    signed_volume=sum(f['normal_sign']*f['plane_um']*(f['u1_um']-f['u0_um'])*(f['v1_um']-f['v0_um']) for f in r['faces'])/3e18
    assert signed_volume==pytest.approx(r['union_volume_m3'],abs=1e-12)


def test_overlapping_unequal_heights_have_only_the_correct_external_step_boundary():
    boxes=[box('a',.5,1.),box('b',1.,2.)];before=deepcopy(boxes);r=wall_union_boundary(boxes)
    assert boxes==before and r['union_volume_m3']==2.5 and r['union_footprint_area_m2']==1.5
    assert r['union_surface_area_m2']==12.
    assert_boundary(boxes,r)


def test_contained_short_box_adds_no_internal_faces_or_volume():
    boxes=[box('short',.5,1.),box('tall',.5,2.)];r=wall_union_boundary(boxes)
    assert r['union_volume_m3']==2. and r['union_surface_area_m2']==10.
    assert_boundary(boxes,r)


def test_uniform_heights_preserve_original_exact_face_contract():
    boxes=[box('a',.5,1.),box('b',1.,1.)]
    old,new=original(boxes),wall_union_boundary(boxes)
    assert all(new[k]==v for k,v in old.items())


def test_all_actual_pilot_specs_and_visual_meshes_construct_before_native_allocation():
    for g in GEOMETRIES:
        spec=next(specification(c) for c in TRIALS if specification(c)['layout_id'].endswith(g))
        boxes=spec['geometry']['wall_boxes'];r=wall_union_boundary(boxes);assert_boundary(boxes,r)
        assert r['compressed_cells']<=65536 and not r['physical_box_geometry_changed']
        with pytest.raises(ValueError,match='common wall'):original(boxes)
        a=independently_seeded_union_surfaces(boxes,'distinctive',spec['appearance_seed'])
        b=independently_seeded_union_surfaces(boxes[::-1],'distinctive',spec['appearance_seed'])
        assert [n for n,_ in a]==['ground_visual','wall_union_visual']
        for (_,ma),(_,mb) in zip(a,b,strict=True):
            np.testing.assert_array_equal(ma.vertices,mb.vertices);np.testing.assert_array_equal(ma.faces,mb.faces)
            np.testing.assert_array_equal(ma.visual.vertex_colors,mb.visual.vertex_colors)
        assert triangle_identity(a[1][1].vertices,a[1][1].faces)['area_m2']==pytest.approx(r['union_surface_area_m2'])


@pytest.mark.parametrize('fault',['rotation','floating','offgrid','duplicate','negative','nonfinite'])
def test_existing_unsupported_geometry_remains_rejected(fault):
    boxes=[box('a',.5,1.),box('b',1.,2.)]
    if fault=='rotation':boxes[0]['yaw_rad']=.1
    if fault=='floating':boxes[0]['centre_xyz'][2]+=.1
    if fault=='offgrid':boxes[0]['centre_xyz'][0]+=1e-7
    if fault=='duplicate':boxes[1]['wall_id']='a'
    if fault=='negative':boxes[0]['size_xyz'][0]=-1.
    if fault=='nonfinite':boxes[0]['size_xyz'][0]=float('nan')
    with pytest.raises(ValueError):wall_union_boundary(boxes)


def test_native_builder_changes_only_the_explicit_visual_surface_provider():
    before=Path('lewm_genesis/lewm_genesis/union_wall_rgbd_scene_development.py').read_text()
    after=Path('lewm_genesis/lewm_genesis/variable_height_union_rgbd_scene_development.py').read_text()
    def functions(s):return [ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)]
    assert functions(before)==functions(after)
