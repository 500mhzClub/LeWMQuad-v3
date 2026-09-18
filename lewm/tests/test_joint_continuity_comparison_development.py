"""Bounded synthetic comparison admission and witness tests; no runtime data."""
import ast
from copy import deepcopy
import inspect
import numpy as np
import pytest

from scripts import run_go2_joint_continuity_comparison_v1 as run
from lewm.joint_continuity_history_verification_development import verify_joint_history
from lewm.independent_tracking_continuity_verification_development import verify_continuity
from lewm.joint_rotation_witness_verification_development import verify_rotations
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture
from lewm.causal_sensor_state import SensorContractError


def test_joint_history_checker_changes_only_declared_mode_role_and_scope_label():
    a = ast.parse(inspect.getsource(verify_continuity)).body[0]
    b = ast.parse(inspect.getsource(verify_joint_history)).body[0]
    a.name = b.name
    a.body[0] = b.body[0]
    for node in ast.walk(a):
        if isinstance(node, ast.Constant):
            if node.value == 'gyro': node.value = 'joint'
            elif node.value == 'rotation_estimator': node.value = 'consistency_monitor_only'
            elif node.value == 'same-frame unpromoted gyro-conditioned pose required':
                node.value = 'same-frame unpromoted joint-RGB-D pose required'
        if isinstance(node, ast.keyword) and node.arg == 'incremental_rotation_witness_saved':
            node.arg = 'incremental_rotation_witness_checked_by_this_function'
    assert ast.dump(a) == ast.dump(b)


def test_negative_bias_is_timestamp_consistent_and_does_not_modify_packet():
    times = np.array([9_899_999_999,9_900_000_000,9_902_000_000],dtype=np.int64)
    channel = dict(measured_ns=times, values=np.array([[1.,2.,3.]]*3))
    p = dict(sensor_state=dict(sensed=dict(gyro=deepcopy(channel))))
    packet = (p,{},deepcopy(channel),9_900_000_000)
    before = deepcopy(packet)
    modified = run.intervention(packet,'gyro_bias_negative',84)
    for a,b in ((modified[0]['sensor_state']['sensed']['gyro'],packet[0]['sensor_state']['sensed']['gyro']),
                (modified[2],packet[2])):
        np.testing.assert_array_equal(a['values'][:,2],[3.,2.98,2.98])
        np.testing.assert_array_equal(b['values'],before[2]['values'])
    next_packet = (p,{},deepcopy(channel),10_000_000_000)
    later = run.intervention(next_packet,'gyro_bias_negative',85)
    np.testing.assert_array_equal(later[2]['values'],modified[2]['values'])


@pytest.fixture
def rows(monkeypatch):
    models = dict(original=run.TemporalAnchorVisualLedMotion(), temporal_anchor=run.JointTemporalAnchorVisualLedMotion())
    for wrapper in models.values():
        model=wrapper.model; candidate=model._candidate
        def deny(ref,current,G,model=model,candidate=candidate):
            if model.frame==2 and ref is not model.previous:
                raise SensorContractError('synthetic retained-reference denial')
            return candidate(ref,current,G)
        monkeypatch.setattr(model,'_candidate',deny)
    result=[]
    for frame,packet in enumerate(packets([texture()]*4)):
        # Synthetic fixture begins at a different timestamp. Shift all clocks
        # consistently before observer inference to the fixed challenge start.
        delta=1_500_000_000-packet[3]+frame*100_000_000
        def shift(value,key=''):
            if isinstance(value,dict):return {k:shift(v,k) for k,v in value.items()}
            if key.endswith('_ns'):
                if isinstance(value,np.ndarray):return value+delta
                if type(value) is int:return value+delta
            return value
        p,d,f,now=packet
        packet=(shift(p),shift(d),shift(f),now+delta)
        arms={a:run.observe(m,deepcopy(packet)) for a,m in models.items()}
        row=dict(frame=frame,measured_ns=packet[3],arm_meaning=run.ARM_MEANING,
            source_packet_sha256='a'*64,intervened_packet_sha256='a'*64,arms=arms,
            native_pose_input=False,navigation_qualified=False)
        row['availability']=run.stress.category(row);result.append(row)
    return result


def test_joint_saved_rotation_and_full_history_are_independently_checked(rows):
    report=run.summaries(rows)
    assert report['arms']['temporal_anchor']['available']==4
    verify_joint_history(rows,report['continuity'])
    checked=verify_rotations(rows)
    assert checked['qualified_rotation_witnesses_checked']>=3
    assert checked['anchor_increment_rotation_disagreements_recomputed']>=2
    assert not checked['native_truth_used']


@pytest.mark.parametrize('fault',['composition','reference','angle','mode','missing_increment','invented_pose'])
def test_witness_corruption_is_rejected(rows,fault):
    rows=deepcopy(rows);e=rows[-1]['arms']['temporal_anchor']['continuity']
    w=e['rotation_measurement_witnesses'][0]
    if fault=='composition': w['composed_rotation_initial_body_from_current_body'][0][0]=2.
    elif fault=='reference': w['reference_frame']=99
    elif fault=='angle': e['disagreement_rad']+=.001
    elif fault=='mode': w['fitting_mode']='gyro'
    elif fault=='missing_increment':e['incremental_rotation_witness']=None
    else:rows[-1]['arms']['temporal_anchor']['pose']['position_initial_body_m'][0]+=.1
    with pytest.raises(ValueError):verify_rotations(rows)


def test_native_evaluation_cannot_start_before_all_new_sensor_streams_admitted(monkeypatch):
    def reject(*args):raise ValueError('missing new sensor stream')
    monkeypatch.setattr(run,'admit_sensor_population',reject)
    monkeypatch.setattr(run.np,'load',lambda *a,**k:pytest.fail('native arrays before admission'))
    with pytest.raises(ValueError,match='missing new sensor'):
        run.evaluate(None,{}, {}, {})


def test_output_cannot_target_old_root_or_overwrite_completed_stream(tmp_path,monkeypatch):
    monkeypatch.setattr(run.custody,'BASE',tmp_path)
    old=tmp_path/'go2_old_attempt_001';old.mkdir()
    fresh=tmp_path/'go2_new_attempt_001';fresh.mkdir()
    monkeypatch.setattr(run.prior,'INPUT',old)
    with pytest.raises(ValueError,match='distinct comparison'):
        run.Store(old,run.base.TRIALS[0],'nominal')
    store=run.Store(fresh,run.base.TRIALS[0],'nominal')
    name=run.stem(store.trial,store.scenario)+'_estimates.jsonl'
    with store.stream(name) as f:store.append(f,{'missing_pose':None})
    with pytest.raises(FileExistsError):
        with store.stream(name):pass


def test_bad_or_extra_scenario_rejected_before_artifact_access():
    with pytest.raises(ValueError):
        run.stem(run.base.TRIALS[0], 'not_a_scenario')
