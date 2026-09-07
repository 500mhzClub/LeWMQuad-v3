from copy import deepcopy

import numpy as np
import pytest

from lewm.action_motion_identification_development import command_schedule
from lewm.action_response_model_development import model_identity
from lewm.action_response_validation_development import MotionValidationController
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_action_motion_identification_development import controller, prediction_fixture
from lewm.tests.test_action_response_model_development import fitted
from lewm.tests.test_continuous_startup_handoff_development import frames
from scripts.audit_go2_action_motion_validation_development_v1 import compare_predictions, audit_outer_timings


def validation():
    model=fitted()
    return MotionValidationController(controller().owner,model,model_identity(model))


def test_B_schedule_complete_with_three_causal_predictions_and_same_memory():
    c=validation(); memory=c.owner._memory; rows=[]
    for p,d,f,t in frames(45):
        row=c.observe(p,d,f,now_ns=t); rows.append(row)
        if row['terminal']: break
    assert c.status=='COMPLETE_VALIDATION_SCHEDULE' and c.owner._memory is memory
    motion=[r for r in rows if r['prediction'] is not None]
    assert len(motion)==28 and [r['requested_command'] for r in motion]==command_schedule('validation')
    for row in motion:
        base=row['prediction']; pos=row['position_persistence_prediction']; learned=row['response_prediction']
        assert base['expected_applied_commands']==pos['expected_applied_commands']==learned['expected_applied_commands']
        assert learned['model_sha256']==c.model_sha256
        assert not learned['navigation_action_permitted'] and not learned['execution_error_validated']
    with pytest.raises(SensorContractError): c.observe(p,d,f,now_ns=t)


def test_model_input_is_copied_and_invalid_identity_rejected():
    m=fitted(); identity=model_identity(m); c=MotionValidationController(controller().owner,m,identity)
    m['body_weights'][0][0]+=100
    assert model_identity(c.model)==identity
    with pytest.raises(SensorContractError): MotionValidationController(controller().owner,m,identity)


def test_finite_bad_predictions_do_not_select_a_different_command():
    first=validation(); m=fitted(); m['body_weights'][0][0]+=100
    second=MotionValidationController(controller().owner,m,model_identity(m))
    for p,d,f,t in frames(8):
        a=first.observe(p,d,f,now_ns=t); b=second.observe(p,d,f,now_ns=t)
    assert a['requested_command']==b['requested_command']==[.1,0.,0.]
    assert a['response_prediction']['positions_current_body_m']!=b['response_prediction']['positions_current_body_m']


def compared_fixture(n=201):
    geometry,raw,decisions,cameras=prediction_fixture(n)
    row=decisions[0]['decision']
    row['position_persistence_prediction']=deepcopy(row['prediction'])
    row['response_prediction']=deepcopy(row['prediction'])
    return geometry,raw,decisions,cameras


def test_matched_metrics_include_true_primitive_centre_error_and_truncation():
    g,raw,d,c=compared_fixture()
    results=compare_predictions(raw,g,d,c)
    assert len(results)==3
    for result in results.values():
        np.testing.assert_allclose([r['maximum_primitive_centre_error_m'] for r in result['rows']],[.01,.02,.03,.04])
        assert [r['scored'] for r in result['summary']]==[1]*4
    g,raw,d,c=compared_fixture(100)
    results=compare_predictions(raw,g,d,c)
    assert [r['status'] for r in results['sensor_response']['rows']]==['SCORED_EXECUTED_FUTURE']+['TRUNCATED_HORIZON']*3


def test_missing_model_forecast_or_changed_command_population_is_rejected():
    g,raw,d,c=compared_fixture()
    d[0]['decision']['response_prediction']=None
    with pytest.raises((ValueError,AssertionError)): compare_predictions(raw,g,d,c)
    g,raw,d,c=compared_fixture()
    d[0]['decision']['response_prediction']['expected_applied_commands'][0][0]=.1
    with pytest.raises((ValueError,AssertionError)): compare_predictions(raw,g,d,c)


def timing_fixture():
    rows=[dict(decision_index=i,captures_before=1,start_perf_counter_ns=i*100_000_000,
        end_perf_counter_ns=i*100_000_000+20_000_000,outer_wall_ms=20.,observation_index=i,
        decision_recorded=True,command_tick_attempted=True,completed_without_exception=True,
        fresh_capture_inside_loop=i>0) for i in range(2)]
    return dict(outer=rows,controller=[dict(controller_wall_ms=4.)]*2,
        captures=[dict(acquisition_and_depth_observer_ms=10.)]*2),[
        dict(observation_index=i) for i in range(2)],[dict(decision_index=i,execution_wall_ms=5.) for i in range(2)]


def test_outer_timing_excludes_initial_precapture_and_contains_components():
    timings,decisions,tape=timing_fixture()
    assert audit_outer_timings(timings,decisions,tape)==[20.]
    timings['outer'][1]['outer_wall_ms']=18.
    timings['outer'][1]['end_perf_counter_ns']=118_000_000
    with pytest.raises((ValueError,AssertionError)): audit_outer_timings(timings,decisions,tape)


@pytest.mark.parametrize('fault',['clock','fresh','missing','nan'])
def test_outer_timing_rejects_invalid_or_incomplete_records(fault):
    timings,decisions,tape=timing_fixture()
    if fault=='clock': timings['outer'][1]['start_perf_counter_ns']=-1
    if fault=='fresh': timings['outer'][0]['fresh_capture_inside_loop']=True
    if fault=='missing': timings['outer'].pop()
    if fault=='nan': timings['outer'][1]['outer_wall_ms']=float('nan')
    with pytest.raises((ValueError,AssertionError)): audit_outer_timings(timings,decisions,tape)
