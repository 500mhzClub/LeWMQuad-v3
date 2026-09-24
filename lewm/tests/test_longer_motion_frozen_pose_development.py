from copy import deepcopy

import numpy as np
import pytest

from scripts.probe_go2_longer_motion_frozen_pose_development_v1 import coverage_summary, initial_up


def test_coverage_empty_or_missing_is_not_full():
    assert coverage_summary([])['first_full_coverage_frame'] is None
    assert coverage_summary([dict(frame=0,floor_coverage={})])['full_coverage_frames']==0


def test_coverage_counts_actual_shapes_and_frames():
    rows=[dict(frame=3,floor_coverage={'a':True,'b':False}),
          dict(frame=8,floor_coverage={'a':True,'b':True})]
    assert coverage_summary(rows)==dict(queries=2,maximum_covered_shapes=2,
        first_full_coverage_frame=8,full_coverage_frames=1)


def packet():
    return {'sensor_state':{'sensed':{'specific_force':{'values':np.tile([0.,0.,9.81],(6,1)),
                                                       'valid':np.ones((6,3),bool)}}}}


def test_initial_up_uses_measured_quiet_force_without_native_pose():
    p=packet(); np.testing.assert_array_equal(initial_up(p),[0.,0.,1.])
    p['sensor_state']['sensed']['specific_force']['values'][:]=[9.81,0.,0.]
    np.testing.assert_array_equal(initial_up(p),[1.,0.,0.])


@pytest.mark.parametrize('fault',['invalid','nan','nonquiet'])
def test_initial_up_rejects_missing_or_nonquiet_measurements(fault):
    p=packet(); f=p['sensor_state']['sensed']['specific_force']
    if fault=='invalid': f['valid'][0,0]=False
    elif fault=='nan': f['values'][0,0]=np.nan
    else: f['values']*=0
    with pytest.raises(ValueError): initial_up(p)


def test_score_retains_failures_common_frames_and_coverage_mismatches(tmp_path,monkeypatch):
    import scripts.probe_go2_longer_motion_frozen_pose_development_v1 as runner
    directory=tmp_path/'fit'; directory.mkdir(); poses=np.zeros((752,7)); poses[:,6]=1.
    yaw=.4; R=np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
    poses[749:,:3]=[1.,2.,3.]; poses[749:,5]=np.sin(yaw/2); poses[749:,6]=np.cos(yaw/2)
    poses[750,:3]+=R@np.array([.1,0.,0.]); poses[751,:3]+=R@np.array([.2,0.,0.])
    np.savez(directory/'physics_trace.npz',base_pose_world=poses)
    def read(path,name):
        if path==directory and name=='camera_audit.json':
            return [dict(physical_sample_index=i) for i in (749,750,751)]
        assert path==tmp_path and name=='fit_raw_acquisition_audit_details.json'
        return {'evaluator_only_initial_surface_queries':[
            dict(frame=0,floor_coverage={'a':False}),dict(frame=1,floor_coverage={'a':True}),
            dict(frame=2,floor_coverage={'a':True})]}
    monkeypatch.setattr(runner,'INPUT',tmp_path); monkeypatch.setattr(runner,'read_json',read)
    rows=[]; failure=dict(frame=2,chain=['synthetic rejection'])
    for frame in range(3):
        members={}
        for mode in ('joint','gyro'):
            position=[.1*frame + (.01 if mode=='gyro' and frame==1 else 0),0.,0.]
            members[mode]=dict(status='CONDITIONAL_RIGID_POSE',state=dict(position_initial_body_m=position,
                rotation_initial_body_from_current_body=np.eye(3).tolist()),floor_coverage={'a':frame!=1})
        if frame==2: members['joint']=dict(status='TERMINAL_FAILURE',failure=failure,state=None,floor_coverage=None)
        rows.append(dict(frame=frame,members=members))
    predictions={'trials':{'fit':dict(rows=rows,keyframes={'joint':[{}],'gyro':[{}]})}}
    before=deepcopy(predictions); result=runner.score(predictions)
    assert predictions==before
    assert result['comparisons']['fit']['common_admitted_frames']==2
    assert result['comparisons']['fit']['gyro_maximum_position_error_m']==pytest.approx(.01)
    joint=result['summaries']['fit__joint']; gyro=result['summaries']['fit__gyro']
    assert joint['admitted']==2 and gyro['admitted']==3 and joint['failure']==failure
    assert joint['maximum_position_error_m']<1e-14 and gyro['maximum_orientation_error_rad']<1e-14
    assert joint['estimated_covered_native_uncovered']==1 and joint['estimated_uncovered_native_covered']==1
    assert not result['parameter_fitting'] and not result['validation_used_for_selection']
    assert not joint['error_bounds_calibrated'] and not result['navigation_qualified']
