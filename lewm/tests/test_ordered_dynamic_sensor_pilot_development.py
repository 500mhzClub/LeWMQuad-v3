from copy import deepcopy
from types import SimpleNamespace as NS
import sys
import numpy as np
import pytest
from scripts.ordered_dynamic_pilot_development import RUNS,TRIALS,definitions,artifacts,compare_streams

def test_exact_prospective_population():
    from lewm.independent_layout_inventory_development import build_inventory
    from lewm.independent_layout_collection_development import CollectionInventory
    inv=CollectionInventory(build_inventory());d=definitions(inv)
    assert len(d)==8 and len({c for _,c,_ in RUNS})==4
    assert [r for _,_,r in RUNS]==[0]*4+[1]*4
    for run,c,r in RUNS:
        assert d[run]['specification']==inv.specification(c) and d[run]['repeat']==r
        assert d[run]['specification']['history_kind']=='recent_forward'
    assert {d[r]['specification']['context_kind'] for r in d}=={'junction','near_wall'}

def test_union_artifact_roster_is_explicit_and_complete():
    s=dict(geometry=dict(wall_boxes=[dict(wall_id='test_wall')]))
    r=dict(setup_checked=True,rgbd_frames=2)
    names=artifacts(s,r)
    assert len(names)==len(set(names))
    assert 'visual_meshes/wall_union_visual.ply' in names and 'visual_meshes/test_wall_visual.ply' not in names
    assert 'raster_0000.json' in names and 'raster_0001.json' in names
    assert 'native_depth_0001.npz' in names and 'startup_native_robot_geometry.json' in names

def stream(complete=True):return dict(physics_samples=12,frames=3,schedule_complete=complete,sha256={'a':'1','b':'2'})

def test_partial_repeats_never_count_as_complete():
    assert compare_streams(stream(),stream())['complete_repeatability']
    r=compare_streams(stream(False),stream(False))
    assert r['exact_available_stream'] and not r['complete_repeatability'] and r['status']=='EXACT_PARTIAL_REPLAY'
    assert not compare_streams(None,stream())['exact_available_stream']

@pytest.mark.parametrize('field',['physics_samples','frames','schedule_complete','hash','missing'])
def test_stream_difference_not_silently_dropped(field):
    a=stream();b=deepcopy(a)
    if field=='hash':b['sha256']['a']='3'
    elif field=='missing':del b['sha256']['a']
    elif field=='schedule_complete':b[field]=False
    else:b[field]+=1
    assert not compare_streams(a,b)['exact_available_stream']

def test_missing_pairs_count_as_unavailable_not_as_matches():
    from scripts.audit_go2_ordered_union_dynamic_sensor_pilot_v1 import comparisons
    p,r=comparisons({})
    assert len(p)==len(r)==4
    assert all(x['status']=='UNAVAILABLE_PREFIX' and not x['matched'] for x in p)
    assert all(x['status']=='UNAVAILABLE_STREAM' for x in r)

def test_initializer_precedes_frozen_initializer_and_retains_guards():
    from scripts.ordered_dynamic_session_development import OrderedDynamicSession
    from scripts.ordered_dynamic_physical_init_development import OrderedDynamicPhysicalInit
    from scripts.independent_pulse_context_physical_init_development import PulseContextPhysicalInit
    from scripts.independent_pulse_context_session_development import PulseContextSession
    from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession
    m=OrderedDynamicSession.__mro__
    assert m.index(AttributedSession)<m.index(OrderedDynamicPhysicalInit)<m.index(PulseContextPhysicalInit)
    assert OrderedDynamicSession._sample is PulseContextSession._sample
    assert OrderedDynamicSession.command_tick is PulseContextSession.command_tick
    with pytest.raises(ValueError):OrderedDynamicSession({}, {},None)

@pytest.mark.parametrize('fault',[None,'samples','position','exception','wrong_target'])
def test_precision_readback_restores_native_context(monkeypatch,fault):
    from scripts.ordered_dynamic_session_development import precision_readback
    events=[];state={'fb':11}
    constants=['GL_DRAW_FRAMEBUFFER','GL_DRAW_FRAMEBUFFER_BINDING','GL_SUBPIXEL_BITS','GL_DEPTH_BITS','GL_SAMPLES','GL_SAMPLE_POSITION']
    gl=NS(**{k:k for k in constants})
    def get(k):
        if fault=='exception' and k=='GL_SUBPIXEL_BITS':raise RuntimeError('readback')
        return {'GL_DRAW_FRAMEBUFFER_BINDING':99 if fault=='wrong_target' else state['fb'],
            'GL_SUBPIXEL_BITS':8,'GL_DEPTH_BITS':24,'GL_SAMPLES':0 if fault=='samples' else 4}[k]
    def bind(target,fb):state['fb']=fb;events.append(('bind',fb))
    gl.glGetIntegerv=get;gl.glBindFramebuffer=bind
    gl.glGetMultisamplefv=lambda _,i: [2.,.5] if fault=='position' else [.25,.75]
    monkeypatch.setitem(sys.modules,'OpenGL.GL',gl)
    ctx=NS(make_current=lambda:events.append('current'),make_uncurrent=lambda:events.append('uncurrent'))
    camera=NS(uid='a',_rasterizer=NS(_renderer=ctx,_camera_targets={'a':NS(_main_fb=11,_main_fb_ms=12)}))
    if fault:
        with pytest.raises((ValueError,RuntimeError)):precision_readback(camera)
    else:
        r=precision_readback(camera);assert r['subpixel_bits']==8 and r['rgb_target_samples']==4
    assert events[0]=='current' and events[-1]=='uncurrent'
    assert state['fb']==(99 if fault=='wrong_target' else 11)

def test_capture_adds_witness_without_replacing_rgb_or_depth(monkeypatch,tmp_path):
    import scripts.ordered_dynamic_session_development as mod
    from scripts.startup_raw_sensor_audit_development import read_json
    returned={'timestamp_s':1.5,'rgb_sha256':'saved'}
    monkeypatch.setattr(mod.NearFieldCapture,'capture_fixed_rgb',lambda *a:returned)
    monkeypatch.setattr(mod,'verify_order',lambda camera,order:order)
    monkeypatch.setattr(mod,'precision_readback',lambda camera:{'subpixel_bits':8})
    s=NS(ctx=NS(build=NS(camera=object())),raster_order={'order':'floor_first'},samples=[0]*750)
    assert mod.OrderedDynamicSession.capture_fixed_rgb(s,tmp_path,'rgb_0000') is returned
    r=read_json(tmp_path,'raster_0000.json');assert r['physical_sample_index']==749 and r['precision']['subpixel_bits']==8
    with pytest.raises(FileExistsError):mod.OrderedDynamicSession.capture_fixed_rgb(s,tmp_path,'rgb_0000')
