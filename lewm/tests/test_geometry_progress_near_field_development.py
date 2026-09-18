"""Prospective acquisition/task separation, exact commands and measurement roles."""
from copy import deepcopy
from pathlib import Path
import ast
import numpy as np
import pytest
from lewm import geometry_progress_near_field_development as task
from lewm import geometry_progress_pilot_development as previous
from scripts import audit_go2_geometry_progress_near_field_v1 as audit
from scripts import run_go2_geometry_progress_near_field_v1 as run
from scripts import read_go2_geometry_progress_commands_v1 as exact
from lewm.tests.test_geometry_progress_command_representation_development import recorded_trace


def test_same_actions_geometry_and_gait_pack_with_explicit_new_camera_identity():
    assert task.assignments is previous.assignments and task.decision is previous.decision
    assert task.schedule is previous.schedule and task.progress_outcome is previous.progress_outcome
    for c in task.TRIALS:
        a,b=previous.specification(c),task.specification(c)
        for k in a:
            if k not in ('scene_id','family'):assert a[k]==b[k]
        p,q=previous.pack(a),task.pack(b)
        assert p.static_objects==q.static_objects and p.robot==q.robot and p.physics_randomization==q.physics_randomization
        assert p.camera.near_m==.05 and q.camera.near_m==.005
        assert p.manifest_sha256!=q.manifest_sha256 and p.scene_id!=q.scene_id
        assert not {'action','action_index','command','outcome'} & set(b)
    bad=task.specification(task.TRIALS[0]);bad['render_near_m']=.05
    with pytest.raises(ValueError):task.pack(bad)


@pytest.mark.parametrize('action',task.ACTIONS)
def test_existing_exact_recorder_representation_validator_reused(action):
    assert audit.audit_commands is exact.audit_commands
    audit.audit_commands(*recorded_trace(action),action)


def raster():
    order=dict(order='floor_first',roles=['floor','walls'],surfaces={'floor':{},'walls':{}})
    precision=dict(rgb_target_samples=1,rgb_target_sample_positions=[[.5,.5]],subpixel_bits=8,depth_target_depth_bits=24)
    return dict(physical_sample_index=749,order=order,precision=precision)


@pytest.mark.parametrize('fault',['clock','order','position','count','drift'])
def test_native_raster_witness_mismatch_rejected(fault):
    r=raster();audit.validate_rasters([r],[dict(physical_sample_index=749)])
    rows=[deepcopy(r)];cameras=[dict(physical_sample_index=749)]
    if fault=='clock':rows[0]['physical_sample_index']=799
    if fault=='order':rows[0]['order']['order']='walls_first'
    if fault=='position':rows[0]['precision']['rgb_target_sample_positions'][0][0]=float('nan')
    if fault=='count':rows[0]['precision']['rgb_target_samples']=2
    if fault=='drift':
        rows.append(deepcopy(r));cameras.append(dict(physical_sample_index=749));rows[-1]['precision']['subpixel_bits']=9
    with pytest.raises(ValueError):audit.validate_rasters(rows,cameras)


def reports():
    return [dict(trial=c,**cell,outcome=dict(successful_progress=cell['action']==(
        'left_arc' if cell['geometry']=='left_open' else 'right_arc'),complete_horizon=True,
        physical_stop=None,acquisition_stop=None),setup_admitted=True,targets={},frames=44,
        hard_measurement_failed_frames=[],strict_physical_visibility_pass=True) for c,cell in task.assignments().items()]


def test_prediction_gate_requires_complete_cohort_and_hard_measurements_preserving_strict_failures():
    rows=reports();assert task.measurement_gate(rows)['prediction_design_and_measurement_gate_pass']
    rows[0]['strict_physical_visibility_pass']=False
    r=task.measurement_gate(rows)
    assert r['prediction_design_and_measurement_gate_pass'] and r['strict_depth_failed_cases']==[rows[0]['trial']]
    assert not r['depth_navigation_qualified']
    rows[0]['hard_measurement_failed_frames']=[17]
    assert not task.measurement_gate(rows)['prediction_design_and_measurement_gate_pass']
    with pytest.raises(ValueError):task.measurement_gate(rows[:-1])
    rows=reports();rows[0]['outcome']['complete_horizon']=False;rows[0]['outcome']['physical_stop']='SPEED_STOP'
    assert not task.measurement_gate(rows)['prediction_design_and_measurement_gate_pass']


def test_new_session_uses_reviewed_capture_and_preserves_recording_chain():
    from scripts.geometry_progress_near_field_session_development import GeometryProgressNearFieldSession,GeometryProgressNearFieldPhysicalInit
    from scripts.core_ordered_dynamic_session_development import CoreOrderedDynamicSession
    from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession
    assert GeometryProgressNearFieldSession.capture_fixed_rgb is CoreOrderedDynamicSession.capture_fixed_rgb
    assert GeometryProgressNearFieldSession.__mro__.index(AttributedSession)<GeometryProgressNearFieldSession.__mro__.index(GeometryProgressNearFieldPhysicalInit)
    assert audit.audit_sensors.__module__=='scripts.near_field_sensor_audit_development'


def test_union_and_every_raster_are_in_exact_artifact_roster():
    names=run.artifacts(task.TRIALS[0],dict(setup_checked=True,rgbd_frames=44))
    assert len(names)==len(set(names))
    assert [n for n in names if n.startswith('visual_meshes/')]==['visual_meshes/ground_visual.ply','visual_meshes/wall_union_visual.ply']
    assert len([n for n in names if n.startswith('raster_')])==44
    assert 'raster_0043.json' in names and 'native_depth_0043.npz' in names


def test_exclusive_attempt_refused_before_preflight(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    monkeypatch.setattr(run,'preflight',lambda:pytest.fail('must not reach preflight'))
    with pytest.raises(ValueError,match='exclusive'):run.main()
