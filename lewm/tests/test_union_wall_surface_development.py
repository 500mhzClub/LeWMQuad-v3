"""Analytic union geometry tests; no native rendering or physical collection."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.independent_layout_inventory_development import build_inventory
from lewm_genesis.union_wall_surface_development import wall_union_boundary,rectangle_patch,independently_seeded_union_surfaces
from lewm_genesis.appearance_surface_development import triangle_identity


def box(name,x,y,sx,sy):return dict(wall_id=name,centre_xyz=[x,y,.7],size_xyz=[sx,sy,1.4],yaw_rad=0.)


def no_overlapping_coplanar_faces(faces):
    groups={}
    for f in faces:groups.setdefault((f['axis'],f['plane_um']),[]).append(f)
    for group in groups.values():
        for i,a in enumerate(group):
            for b in group[i+1:]:
                u=min(a['u1_um'],b['u1_um'])-max(a['u0_um'],b['u0_um'])
                v=min(a['v1_um'],b['v1_um'])-max(a['v0_um'],b['v0_um'])
                assert u<=0 or v<=0


def test_overlapping_collinear_boxes_have_only_the_union_boundary():
    boxes=[box('a',0,0,1.28,.08),box('b',1.2,0,1.28,.08)]
    r=wall_union_boundary(boxes)
    assert r['union_footprint_area_m2']==pytest.approx(2.48*.08)
    assert r['union_surface_area_m2']==pytest.approx(2*(2.48*.08+2.48*1.4+.08*1.4))
    no_overlapping_coplanar_faces(r['faces'])


def test_crossing_walls_remove_internal_overlap_faces_without_changing_physical_boxes():
    boxes=[box('a',0,0,1.28,.08),box('b',0,0,.08,1.28)];before=deepcopy(boxes)
    r=wall_union_boundary(boxes);assert boxes==before
    assert r['union_footprint_area_m2']==pytest.approx(2*1.28*.08-.08**2)
    # Plus-shaped footprint perimeter is exactly4*1.28.
    assert r['union_surface_area_m2']==pytest.approx(2*r['union_footprint_area_m2']+4*1.28*1.4)
    no_overlapping_coplanar_faces(r['faces'])


def test_all12_inventory_wall_unions_have_no_duplicate_area_and_outward_boundary_only():
    for layout in build_inventory()['layouts']:
        boxes=layout['wall_boxes'];r=wall_union_boundary(boxes);no_overlapping_coplanar_faces(r['faces'])
        centres=np.array([b['centre_xyz'] for b in boxes]);half=np.array([b['size_xyz'] for b in boxes])/2
        for f in r['faces']:
            axis=f['axis'];j,k=(axis+1)%3,(axis+2)%3;p=np.zeros(3)
            p[axis]=f['plane_um']/1e6;p[j]=(f['u0_um']+f['u1_um'])/2e6;p[k]=(f['v0_um']+f['v1_um'])/2e6
            normal=np.eye(3)[axis]*f['normal_sign']
            inside=np.any(np.all(np.abs(p-1e-7*normal-centres)<half,axis=-1))
            outside=not np.any(np.all(np.abs(p+1e-7*normal-centres)<half,axis=-1))
            assert inside and outside
        assert not r['physical_box_geometry_changed'] and not r['native_visibility_qualified']


def test_rectangle_winding_is_outward_for_all_six_normals():
    r=wall_union_boundary([box('a',0,0,.2,.3)])
    assert len(r['faces'])==6
    for f in r['faces']:
        mesh=rectangle_patch(f,arm='distinctive',seed=1);normal=np.eye(3)[f['axis']]*f['normal_sign']
        np.testing.assert_allclose(mesh.face_normals,np.tile(normal,(len(mesh.faces),1)),atol=1e-12,rtol=0)


def test_geometry_and_appearance_are_order_invariant_and_seed_changes_only_colours():
    boxes=[box('b',1.2,0,1.28,.08),box('a',0,0,1.28,.08)]
    a=independently_seeded_union_surfaces(boxes,'distinctive',19)
    b=independently_seeded_union_surfaces(boxes[::-1],'distinctive',19)
    c=independently_seeded_union_surfaces(boxes,'distinctive',20)
    for (na,ma),(nb,mb),(nc,mc) in zip(a,b,c,strict=True):
        assert na==nb==nc
        np.testing.assert_array_equal(ma.vertices,mb.vertices);np.testing.assert_array_equal(ma.faces,mb.faces)
        np.testing.assert_array_equal(ma.visual.vertex_colors,mb.visual.vertex_colors)
        assert triangle_identity(ma.vertices,ma.faces)==triangle_identity(mc.vertices,mc.faces)
        assert not np.array_equal(ma.visual.vertex_colors,mc.visual.vertex_colors)


@pytest.mark.parametrize('fault',['rotation','unequal_height','floating_base','off_grid','duplicate','negative','nan'])
def test_unsupported_geometry_is_rejected_not_silently_approximated(fault):
    boxes=[box('a',0,0,1.28,.08),box('b',1.2,0,1.28,.08)]
    if fault=='rotation':boxes[0]['yaw_rad']=.1
    elif fault=='unequal_height':boxes[0]['size_xyz'][2]=1.2
    elif fault=='floating_base':
        for b in boxes:b['centre_xyz'][2]=1.
    elif fault=='off_grid':boxes[0]['centre_xyz'][0]+=2e-7
    elif fault=='duplicate':boxes[1]['wall_id']='a'
    elif fault=='negative':boxes[0]['size_xyz'][0]=-1
    else:boxes[0]['centre_xyz'][0]=float('nan')
    with pytest.raises(ValueError):wall_union_boundary(boxes)
