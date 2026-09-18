"""Prospective comparison semantics; no recorded-history or native execution."""
from copy import deepcopy

import pytest
import torch

from scripts import extended_return_budget_prefix_comparison_development as check
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.extended_return_budget_mission_development import ExtendedReturnBudgetMeasuredMission
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMission
from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def controllers():
    return [cls(FixedHeadModel('direct'), ArticulatedCollisionGeometry(URDF),
        public_mission=deepcopy(MISSION), navigation_ticks=budget,
        condition='direct', variant='no_rgb', persistent=True)
        for cls,budget in ((check.candidate.MeasuredPlaneChainedSinglePassController,4000),
            (check.candidate.ExtendedReturnBudgetChainedController,8000))]


@pytest.fixture(scope='module')
def actual_short():
    arms=controllers(); snapshots=[{k:v.clone() for k,v in c.model.state_dict().items()} for c in arms]
    tracker=check.PrefixComparison(); rows=[]
    assert check.fingerprint(check.observed_state(arms[0])) == check.fingerprint(check.observed_state(arms[1]))
    for frame,source in enumerate(sequence()):
        decisions=[];calls=[]
        for controller in arms:
            before=len(controller.model.calls)
            p,d,f,kwargs=[move_test_origin(deepcopy(v)) for v in source]
            decisions.append(controller.observe(p,d,f,**kwargs))
            calls.append(len(controller.model.calls)-before)
        baseline,extended=decisions;recorded=check.to_recorded(baseline)
        row=tracker.observe(baseline,extended,recorded,model_calls=calls)
        assert row['complete_normalized_decision_exact'] and not row['stop']
        assert check.fingerprint(check.observed_state(arms[0])) == check.fingerprint(check.observed_state(arms[1]))
        rows.append((deepcopy(baseline),deepcopy(extended),deepcopy(recorded),calls))
    assert sum(row[3][0] for row in rows)==1
    for controller,snapshot in zip(arms,snapshots,strict=True):
        assert all(torch.equal(snapshot[k],v) for k,v in controller.model.state_dict().items())
        assert all(p.grad is None for p in controller.model.parameters())
    assert check.fingerprint(check.observer_state_tree(arms[0].geometry)) == check.fingerprint(check.observer_state_tree(arms[1].geometry))
    return rows


def test_real_short_sensor_action_sequence_with_different_budgets(actual_short):
    assert len(actual_short)==4
    for baseline,extended,recorded,_ in actual_short:
        assert baseline['shared_navigation_budget_ticks']==4000
        assert extended['shared_navigation_budget_ticks']==8000
        assert check.normalize(extended)==baseline
        assert check.to_recorded(baseline)==recorded


def test_full_synthetic_mission_prefix_stops_at_actual_terminal_intervention(actual_short):
    # Full mission and comparison population, using a warmup decision template.
    # This is not a full sensor/controller replay and supplies no physical path.
    templates=actual_short[0][:2]
    missions=[cls(deepcopy(MISSION),navigation_ticks=budget) for cls,budget in (
        (MeasuredFloorTransportMission,4000),(ExtendedReturnBudgetMeasuredMission,8000))]
    tracker=check.PrefixComparison()
    with pytest.raises(ValueError,match='stopping'):tracker.report()
    for frame in range(4004):
        decisions=[]
        for template,mission in zip(templates,missions,strict=True):
            receipt=mission.advance([.5,.5,0.],frame=frame,now_ns=1_500_000_000+100_000_000*frame,
                previous_requested_command=[0.,0.,0.])
            decision=deepcopy(template)
            decision.update(tick=frame,mission_receipt=receipt,terminal=receipt['terminal'])
            decisions.append(decision)
        row=tracker.observe(*decisions,check.to_recorded(decisions[0]),model_calls=[0,0])
        assert row['stop'] is (frame==4003)
        assert row['complete_normalized_decision_exact'] is (frame<4003)
    report=tracker.report()
    assert report['frames']==4004 and report['budget_only_preboundary_decisions_supported']
    assert report['boundary']['terminal_changed'] and not report['boundary']['requested_command_changed']
    assert not report['following_intervention_observations_consumed']
    assert not report['physical_prefix_verified'] and not report['verified_round_trip']
    with pytest.raises(ValueError,match='no observation'):
        tracker.observe(*decisions,check.to_recorded(decisions[0]),model_calls=[0,0])


@pytest.mark.parametrize('fault', ['unknown_evidence','nested_budget','command','terminal'])
def test_any_early_change_stops_and_cannot_support_budget_only_behavior(actual_short,fault):
    old,new,recorded,calls=deepcopy(actual_short[0])
    if fault=='unknown_evidence':new['unexpected_tracking_evidence']={'accepted':True}
    elif fault=='nested_budget':new['unexpected']={'global_navigation_ticks':8000}
    elif fault=='command':new['requested_command']=[.2,0.,0.]
    else:new['terminal']='SENSOR_OR_MODEL_FAILURE'
    tracker=check.PrefixComparison();row=tracker.observe(old,new,recorded,model_calls=calls)
    assert row['stop'] and not row['complete_normalized_decision_exact']
    assert not tracker.report()['budget_only_preboundary_decisions_supported']
    with pytest.raises(ValueError,match='no observation'):
        tracker.observe(*actual_short[1][:3],model_calls=actual_short[1][3])


@pytest.mark.parametrize('fault', ['recorded','tick','boolean_tick','budget','boolean_budget','receipt_budget',
    'identity','flag','calls','invented_forecast'])
def test_changed_reference_identity_budget_or_model_evidence_is_rejected(actual_short,fault):
    old,new,recorded,calls=deepcopy(actual_short[0])
    if fault=='recorded':recorded['unexpected']=True
    elif fault=='tick':new['tick']=1
    elif fault=='boolean_tick':new['tick']=False
    elif fault=='budget':new['shared_navigation_budget_ticks']=4000
    elif fault=='boolean_budget':new['shared_navigation_budget_ticks']=True
    elif fault=='receipt_budget':new['mission_receipt']['global_navigation_ticks']=4000
    elif fault=='identity':new['controller']='other'
    elif fault=='flag':new['extended_return_budget_enabled']=False
    elif fault=='calls':calls=[0,True]
    else:new['new_selection']={'prediction':[1.]}
    with pytest.raises(ValueError):check.PrefixComparison().observe(old,new,recorded,model_calls=calls)


def test_state_normalization_retains_unknown_fields_motion_and_nonbudget_mission_state():
    baseline,extended=controllers()
    expected=check.fingerprint(check.observed_state(baseline))
    assert check.fingerprint(check.observed_state(extended))==expected
    extended.motion.unknown_witness={'global_navigation_ticks':8000}
    assert check.fingerprint(check.observed_state(extended))!=expected
    del extended.motion.unknown_witness
    extended.mission.quiet=1
    assert check.fingerprint(check.observed_state(extended))!=expected
    extended.mission.quiet=0
    extended.unknown={'type':check.candidate.ExtendedReturnBudgetMemory.__module__+'.ExtendedReturnBudgetMemory',
        'fields':{'global_navigation_ticks':8000}}
    assert check.observed_state(extended)['unknown']==extended.unknown
    del extended.unknown
    assert check.fingerprint(check.observed_state(extended))==expected
    extended.mapper.surface=deepcopy(extended.memory)
    with pytest.raises(ValueError,match='aliases'):check.observed_state(extended)
