"""No simulation: exact scene compilation and no-privilege collection selector."""
from copy import deepcopy
import math
import numpy as np
import pytest
from lewm.independent_layout_inventory_development import build_inventory
from lewm.independent_layout_collection_development import CollectionInventory,schedule,decision
from lewm.tests.test_independent_pulse_context_development import policy


def test_all1440_scenes_compile_exact_inventory_geometry_without_role_changes():
    inv=CollectionInventory(build_inventory());covered=[]
    for batch in inv.batches:
        ids=inv.episode_ids(batch);assert len(ids)==120
        for eid in ids:
            spec=inv.specification(eid);p=inv.pack(spec);covered.append(eid)
            assert p.scene_id==spec['scene_id'] and len(p.static_objects)==len(spec['geometry']['wall_boxes'])
            assert p.physics_randomization.floor_friction_mu==spec['friction_mu']
            assert p.robot.spawn_xyz_m==tuple(spec['spawn_se2_world'][:2]+[.375])
            yaw=spec['spawn_se2_world'][2]
            np.testing.assert_allclose(p.robot.spawn_quat_wxyz,[math.cos(yaw/2),0.,0.,math.sin(yaw/2)],rtol=0,atol=0)
            assert spec['data_role']==spec['evaluation_layout']['role']==spec['role']
            assert p.physics_seed==spec['physics_seed'] and p.visual_seed==spec['appearance_seed']
            assert p.camera.near_m==spec['render_near_m']==.005 and spec['public_depth_range_m']==[.2,5.]
            assert p.camera.xyz_body_m==(.326,0.,.043) and p.camera.far_m==200.
            for actual,wanted in zip(p.static_objects,spec['geometry']['wall_boxes'],strict=True):
                assert actual.object_id==wanted['wall_id'] and actual.center_xyz_m==tuple(wanted['centre_xyz'])
                assert actual.size_xyz_m==tuple(wanted['size_xyz'])
    assert len(set(covered))==1440
    with pytest.raises(ValueError):inv.episode_ids('l12')


@pytest.mark.parametrize('fault',['friction','role','spawn','wall','bool'])
def test_mutated_scene_cannot_reuse_frozen_identity(fault):
    inv=CollectionInventory(build_inventory());s=inv.specification(inv.episode_ids('l00')[1])
    if fault=='friction':s['friction_mu']=.7
    elif fault=='role':s['data_role']='development_eval'
    elif fault=='spawn':s['geometry']['spawn_se2_world'][0]+=.1
    elif fault=='wall':s['geometry']['wall_boxes'].pop()
    elif fault=='bool':s['action_index']=True
    with pytest.raises(ValueError):inv.pack(s)


@pytest.mark.parametrize('a',range(6))
def test_quiet_recent_factor_changes_only_common_history_not_candidate(a):
    quiet=schedule(a,'quiet');recent=schedule(a,'recent_forward')
    assert quiet[8:]==recent[8:]
    assert all(r['requested_command']==[0.,0.,0.] for r in quiet[:8])
    assert all(r['requested_command']==[.12,0.,0.] for r in recent[:8])
    for kind in ('quiet','recent_forward'):
        d=decision(a,kind,8,policy(8));assert not d['tracker_required'] and not d['native_state_used']
        assert d['requested_command']==quiet[8]['requested_command']
        assert decision(a,kind,len(quiet),policy(len(quiet)))['terminal']
    with pytest.raises(ValueError):decision(a,'quiet',9,policy(8))


def test_selector_contract_has_no_geometry_friction_or_target_input():
    import inspect
    assert tuple(inspect.signature(decision).parameters)==('action_index','history_kind','tick','policy')
    p=policy(8);p['friction_mu']=.15
    with pytest.raises(ValueError):decision(0,'quiet',8,p)
    with pytest.raises(ValueError):schedule(0,'unknown')
    with pytest.raises(ValueError):schedule(True,'quiet')


def test_returned_specs_cannot_mutate_another_scene_or_the_input_manifest():
    original=build_inventory();before=deepcopy(original);inv=CollectionInventory(original)
    eid=inv.episode_ids('l00')[0];s=inv.specification(eid);s['geometry']['wall_boxes'][0]['centre_xyz'][0]+=100
    assert inv.specification(eid)['geometry']['wall_boxes'][0]['centre_xyz'][0]!=s['geometry']['wall_boxes'][0]['centre_xyz'][0]
    assert original==before


def test_inventory_initializer_precedes_frozen_pilot_initializer_in_full_recorder_chain():
    from scripts.independent_layout_session_development import InventorySession
    from scripts.independent_layout_physical_init_development import InventoryPhysicalInit
    from scripts.independent_pulse_context_physical_init_development import PulseContextPhysicalInit
    from scripts.independent_pulse_context_session_development import PulseContextSession
    from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession
    mro=InventorySession.__mro__
    assert mro.index(AttributedSession)<mro.index(InventoryPhysicalInit)<mro.index(PulseContextPhysicalInit)
    assert InventorySession._sample is PulseContextSession._sample
    assert InventorySession.sensor_packets is PulseContextSession.sensor_packets
    assert InventorySession.command_tick is PulseContextSession.command_tick
    from scripts.near_field_rgbd_capture_development import NearFieldCapture
    assert InventorySession.capture_fixed_rgb is NearFieldCapture.capture_fixed_rgb
    assert InventorySession.capture_fixed_rgb is not PulseContextSession.capture_fixed_rgb


def test_inventory_session_rejects_unvalidated_constructor_before_native_access(tmp_path):
    from scripts.independent_layout_session_development import InventorySession
    with pytest.raises(ValueError):InventorySession({}, {},tmp_path)
