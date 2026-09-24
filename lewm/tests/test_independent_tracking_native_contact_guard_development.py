"""Actual new guard and inherited recorders, with synthetic solver state only."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts import independent_tracking_native_contact_guard_development as mod
from lewm.tests.independent_tracking_native_contact_fixtures import runtime,report_for_count,MockCollectionGuard
from lewm.tests.test_independent_tracking_recording_budget_development import synthetic_native_session
from lewm.tests.test_independent_tracking_collection_development import store,native_mock
from lewm.tests.test_independent_tracking_cohort_development import collected


@pytest.mark.parametrize('fault',[None,'missing','changed','symlink'])
def test_exact_native_source_bindings_reject_changed_or_redirected_implementation(tmp_path,monkeypatch,fault):
    root=tmp_path/'synthetic_native';root.mkdir()
    bindings={}
    for name in mod.NATIVE_FILES:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        data=('synthetic source identity '+name).encode();p.write_bytes(data)
        bindings[name]=hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(mod,'NATIVE_ROOT',root);monkeypatch.setattr(mod,'NATIVE_FILES',bindings)
    if fault=='missing':(root/'__init__.py').unlink()
    elif fault=='changed':(root/'__init__.py').write_bytes(b'changed synthetic implementation')
    elif fault=='symlink':
        (root/'__init__.py').unlink();(root/'__init__.py').symlink_to(root/'options/solvers.py')
    if fault:
        with pytest.raises(ValueError):mod.native_source_bindings()
    else:assert mod.native_source_bindings()=={str(root/n):h for n,h in bindings.items()}


def test_challenge_source_verifier_requires_native_contact_binding_but_does_not_change_predecessor_schema(monkeypatch):
    from scripts import run_go2_independent_tracking_challenge_v1 as launcher
    seen=[];expected={'synthetic-native-source':'a'*64}
    monkeypatch.setattr(launcher,'verify_ordered_source',lambda d:seen.append(d))
    monkeypatch.setattr(launcher,'native_source_bindings',lambda:expected)
    d=dict(schema='independent_tracking_complete_native_challenge.v1',native_contact_source_sha256=expected)
    launcher.verify_ordered_launch(d)
    with pytest.raises(ValueError):launcher.verify_ordered_launch(dict(schema=d['schema']))
    launcher.verify_ordered_launch(dict(schema='unchanged_original_learning_fixture'))
    assert len(seen)==3


@pytest.mark.parametrize('fault',[None,'backend','float64','int64','grad','collision','self_collision',
    'box','environments','option','effective','possible','pairs','contacts','per_pair'])
def test_actual_effective_capacity_fields_are_validated(fault):
    solver,gs,state=runtime();info=solver.collider._collider_info
    if fault=='backend':gs.backend='gpu_fixture'
    elif fault=='float64':gs.np_float=np.float64
    elif fault=='int64':gs.np_int=np.int64
    elif fault=='grad':solver._static_rigid_sim_config.requires_grad=True
    elif fault=='collision':solver._options.enable_collision=False
    elif fault=='self_collision':solver._options.enable_self_collision=False
    elif fault=='box':solver._options.box_box_detection=True
    elif fault=='environments':solver.n_envs=2
    elif fault=='option':solver._options.max_collision_pairs=149
    elif fault=='effective':solver.max_collision_pairs=149
    elif fault=='possible':info.max_possible_pairs[None]=0
    elif fault=='pairs':info.max_collision_pairs[None]=149
    elif fault=='contacts':info.max_contact_pairs[None]=751
    elif fault=='per_pair':solver.collider._collider_static_config.n_contacts_per_pair=16
    if fault:
        with pytest.raises(ValueError):mod.NativeContactGuard(solver,gs)
        assert state.calls==0
    else:
        guard=mod.NativeContactGuard(solver,gs)
        assert state.calls==1 and guard.initial['allocated_contacts']==750
        guard.finish(0);assert mod.validate_report(guard.report(),0)['native_error_check_coverage_verified']


def test_smaller_actual_pair_allocation_is_reported_not_replaced_by_coarse_bound():
    solver,gs,_=runtime();info=solver.collider._collider_info
    info.max_possible_pairs[None]=80;info.max_collision_pairs[None]=80;info.max_contact_pairs[None]=400
    guard=mod.NativeContactGuard(solver,gs);guard.finish(0)
    assert guard.report()['initial_contract']['allocated_contacts']==400
    mod.validate_report(guard.report(),0)


def test_every_sample_and_terminal_call_check_errno_without_waiting_for_native_period():
    solver,gs,state=runtime();guard=mod.NativeContactGuard(solver,gs)
    for n in range(13):guard.before_sample(n)
    guard.finish(13)
    assert state.calls==15  # initial + every sample + terminal, not every tenth.
    row=guard.report();mod.validate_report(row,13)
    assert row['sample_checks_passed']==13 and row['failure'] is None
    with pytest.raises(ValueError):guard.before_sample(13)
    with pytest.raises(ValueError):guard.finish(13)


@pytest.mark.parametrize('stage',['initial','sample','terminal'])
def test_errno_failures_are_latched_and_cannot_be_cleared_into_success(stage):
    solver,gs,state=runtime()
    if stage=='initial':
        state.error='native contact capacity overflow'
        with pytest.raises(RuntimeError):mod.NativeContactGuard(solver,gs)
        return
    guard=mod.NativeContactGuard(solver,gs);guard.before_sample(0)
    state.error='native contact capacity overflow'
    with pytest.raises(RuntimeError):
        if stage=='sample':guard.before_sample(1)
        else:guard.finish(1)
    first=deepcopy(guard.report()['failure']);state.error=None
    with pytest.raises(ValueError):guard.before_sample(1)
    if stage=='sample':
        with pytest.raises(ValueError):guard.finish(1)
    assert guard.report()['failure']==first and not guard.report()['terminal_check_passed']
    with pytest.raises(ValueError):mod.validate_report(guard.report(),1)


@pytest.mark.parametrize('stage',['sample','terminal'])
def test_valid_but_changed_capacity_contract_is_rejected(stage):
    solver,gs,_=runtime();guard=mod.NativeContactGuard(solver,gs)
    info=solver.collider._collider_info
    info.max_possible_pairs[None]=100;info.max_collision_pairs[None]=100;info.max_contact_pairs[None]=500
    with pytest.raises(ValueError):
        if stage=='sample':guard.before_sample(0)
        else:guard.finish(0)
    assert guard.report()['failure'] is not None


def test_missing_or_repeated_recording_positions_cannot_resume():
    solver,gs,_=runtime();guard=mod.NativeContactGuard(solver,gs)
    with pytest.raises(ValueError):guard.before_sample(1)
    with pytest.raises(ValueError):guard.before_sample(0)
    with pytest.raises(ValueError):guard.finish(0)
    solver,gs,_=runtime();guard=mod.NativeContactGuard(solver,gs);guard.before_sample(0)
    with pytest.raises(ValueError):guard.finish(0)  # Guard passed, but getter never produced a row.
    assert not guard.report()['terminal_check_passed']


def test_overflow_prevents_actual_inherited_recorder_from_accepting_current_step(store):
    s,state,sample=synthetic_native_session(store.directory)
    sample();before=(len(s.samples),len(s.packets),len(s.fast_rows))
    def overflow():raise RuntimeError('overflow in just-completed native step')
    s.contact_integrity.solver.check_errno=overflow
    def forbidden():raise AssertionError('overflowed step must not reach pose/contact acquisition')
    s.ctx.build.robot.get_pos=forbidden
    with pytest.raises(RuntimeError,match='overflow'):sample()
    assert before==(len(s.samples),len(s.packets),len(s.fast_rows))==(1,1,1)
    assert s.contact_integrity.report()['sample_checks_attempted']==2
    assert s.contact_integrity.report()['sample_checks_passed']==1


def test_physical_stop_row_is_retained_when_native_errno_is_clean(store):
    from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
    s,state,sample=synthetic_native_session(store.directory);state.stop=True
    with pytest.raises(PhysicalStop):sample()
    assert len(s.samples)==len(s.packets)==len(s.fast_rows)==1
    s.contact_integrity.finish(1)
    mod.validate_report(s.contact_integrity.report(),1)
    assert s.contact_events[-1]['disallowed_contacts']  # Still a physical negative, not completion.


def test_constructor_guard_failure_destroys_scene_once_before_returning_session(monkeypatch,tmp_path):
    from scripts import independent_tracking_session_development as session
    from lewm.independent_tracking_challenge_development import specification,TRIALS
    calls=[]
    def init(self,*args):self.ctx=NS(build=NS(scene=NS(destroy=lambda:calls.append('destroy'))))
    monkeypatch.setattr(session.PulseContextSession,'__init__',init)
    def failed(_):raise RuntimeError('initial errno failure')
    monkeypatch.setattr(session,'from_scene',failed)
    with pytest.raises(RuntimeError):session.IndependentTrackingSession(specification(TRIALS[0]),tmp_path)
    assert calls==['destroy']


def test_terminal_errno_failure_is_not_hidden_by_a_physical_stop(store,native_mock,monkeypatch):
    from scripts.independent_tracking_collection_development import collect_episode
    from lewm.independent_tracking_challenge_development import specification,TRIALS
    native_mock.fault='command'
    def fail(self,n):
        self.row=report_for_count(n);self.row['terminal_check_passed']=False
        self.row['failure']=dict(stage='terminal',error='synthetic overflow',before_recorded_sample=n)
        raise RuntimeError('synthetic final errno overflow')
    monkeypatch.setattr(MockCollectionGuard,'finish',fail)
    result,receipt=collect_episode(store,specification(TRIALS[0]),'a'*64)
    assert result['status']=='TERMINAL_TRACKING_COLLECTION_INFRASTRUCTURE_FAILURE'
    assert result['physical_stop']=='NATIVE_CONTACT_STOP'
    assert result['secondary_failures'][0]['stage']=='terminal_contact_integrity'
    assert not result['native_contact_integrity']['terminal_check_passed']
    assert {'physics_trace.npz','command_tape.json','failure.json'}<=set(receipt['artifact_sha256'])
    assert native_mock.events[-2:]==['destroy','shutdown']


@pytest.mark.parametrize('fault',['missing','failure','count','terminal','capacity'])
def test_rebound_invalid_integrity_metadata_cannot_enter_cohort(collected,fault):
    from scripts import independent_tracking_cohort_development as cohort
    store,_,_=collected;trial=cohort.TRIALS[0];row=deepcopy(store.episodes[trial]);r=row['result']
    if fault=='missing':r.pop('native_contact_integrity')
    elif fault=='failure':r['native_contact_integrity']['failure']=dict(error='overflow')
    elif fault=='count':r['native_contact_integrity']['sample_checks_passed']-=1
    elif fault=='terminal':r['native_contact_integrity']['terminal_check_passed']=False
    else:r['native_contact_integrity']['terminal_contract']['allocated_contacts']=751
    path=store.output/trial/'result.json';path.write_bytes(cohort.encode(r))
    row['receipt']['artifact_sha256']['result.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    row['receipt']['artifact_sizes']['result.json']=path.stat().st_size
    row['receipt']['artifact_bytes']=sum(row['receipt']['artifact_sizes'].values())
    with pytest.raises((ValueError,KeyError)):
        cohort.verify_episode(store.output,trial,r,row['receipt'])
