import numpy as np
import pytest

from lewm.visual_course_response_development import VisualCourseWindow,action_features,fit_response
from lewm.course_aware_visual_servo_development import CourseAwareVisualServo
from lewm.tests.test_bounded_visual_servo_development import evidence


def test_course_measures_lateral_motion_separately_from_yaw():
    model=VisualCourseWindow()
    for i in range(6):
        result=model.observe(measured_ns=1_500_000_000+i*100_000_000,position_initial_xy_m=[i*.003,i*.003],yaw_rad=0.)
    assert result['status']=='COURSE_AVAILABLE'
    np.testing.assert_allclose(result['velocity_initial_xy_m_s'],[.03,.03],atol=1e-14)
    assert result['current_course_hypothesis_rad']==pytest.approx(np.pi/4)
    assert result['physical_velocity_error_bound_m_s'] is None


def test_course_incomplete_low_speed_and_clock_gap_are_explicit():
    model=VisualCourseWindow()
    for i in range(6):
        r=model.observe(measured_ns=i*100_000_000,position_initial_xy_m=[0.,0.],yaw_rad=0.)
        assert r['status']==('COURSE_WINDOW_INCOMPLETE' if i<5 else 'COURSE_LOW_SPEED')
        assert r['current_course_hypothesis_rad'] is None
    with pytest.raises(ValueError): model.observe(measured_ns=700_000_000,position_initial_xy_m=[0,0],yaw_rad=0.)


def test_heading_unwrap_does_not_make_spurious_full_rotation():
    model=VisualCourseWindow()
    for i in range(6):
        yaw=np.arctan2(np.sin(3.12+i*.01),np.cos(3.12+i*.01))
        r=model.observe(measured_ns=i*100_000_000,position_initial_xy_m=[-.003*i,0.],yaw_rad=yaw)
    assert r['mean_yaw_rate_rad_s']==pytest.approx(.1)
    assert abs(r['body_course_offset_rad'])<.02


def test_course_feedback_corrects_more_than_body_heading_only():
    model=CourseAwareVisualServo()
    for i in range(6):
        now=1_500_000_000+i*100_000_000; r=model.step(evidence(now,x=.003*i,y=.003*i),now_ns=now)
    assert r['diagnostic']['course_used']
    assert r['diagnostic']['steering_error_rad']<-.7
    assert r['requested_command'][2]==-.25
    assert not r['supervised_response_fit_used_for_commands']


def test_course_controller_completes_same_full_sequence_and_stops_on_stale():
    model=CourseAwareVisualServo(); now=1_500_000_000
    model.step(evidence(now),now_ns=now)
    for _ in range(11):
        now+=100_000_000; r=model.step(evidence(now,x=.4),now_ns=now)
    assert r['stage']=='turn'
    for _ in range(11):
        now+=100_000_000; r=model.step(evidence(now,x=.4,yaw=.3),now_ns=now)
    assert r['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE'
    other=CourseAwareVisualServo(); bad=evidence(now);bad['current_pose']=None
    assert other.step(bad,now_ns=now)['terminal']=='VISUAL_SERVO_FAILED'


def test_supervised_response_fit_is_finite_and_reports_unidentified_design():
    rng=np.random.default_rng(123); X=rng.normal(size=(100,7)); X[:,0]=1.
    B=rng.normal(size=(7,3)); Y=X@B
    fitted=fit_response(X,Y)
    assert fitted['standardized_rank']==7 and not fitted['causal_action_effect_identified']
    np.testing.assert_allclose(X@fitted['coefficients'],Y,atol=.004)
    X=np.zeros((10,7)); X[:,0]=1.
    fitted=fit_response(X,np.ones((10,3)))
    assert fitted['action_design_rank']==1 and fitted['standardized_rank']==1


def test_response_features_use_proposed_command_and_past_motion():
    course=dict(velocity_initial_xy_m_s=[0.,1.],mean_yaw_rate_rad_s=.2)
    row=action_features([.1,0.,-.25],course,np.pi/2)
    np.testing.assert_allclose(row,[1.,.1,-.25,-.025,1.,0.,.2],atol=1e-14)
