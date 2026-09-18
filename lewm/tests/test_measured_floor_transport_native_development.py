"""Source scope and synthetic intervention admission, not a physical experiment."""
import ast
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import pytest
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_measured_floor_transport_development import transported
from lewm.measured_floor_transport_development import transport_evidence
from scripts import measured_floor_transport_intervention_development as admission
from scripts.run_go2_measured_floor_transport_maze_pilot_v1 import admit_renderer, NATIVE_RESULT


def prefix_report():
    return dict(status='MEASURED_FLOOR_TRANSPORT_PREFIX_COMPLETE', frames=1905, first_intervention_frame=1904,
        complete_preintervention_decisions_exact_outside_validated_labels=True,
        all_1904_prior_actual_commands_exact=True, raw_visual_evidence_exact_at_intervention=True,
        current_transport_pose_available=True, active_return_at_intervention=True, model_state_unchanged=True,
        public_input_arrays_unchanged=True, diagnosis_bindings_match_final_native_artifacts=True,
        following_recorded_observations_consumed=False, native_execution=False,
        final_requested_command=[0., 0., -.45])


def candidate(frame, evidence=None):
    return dict(controller='measured_floor_transport_round_trip_controller_v1',
        floor_transport_during_missingness_enabled=True, terminal=None, failure=None,
        requested_command=[0., 0., -.45] if frame == 1904 else [0., 0., 0.],
        mission_receipt=dict(phase='RETURN', observed_settling=None), original_visual_evidence=None,
        evidence=evidence or dict(schema='joint_floor_registered_pose_evidence_development.v1', current_pose=dict(frame=frame)))


def shifted(value):
    if isinstance(value, list): return [shifted(v) for v in value]
    if not isinstance(value, dict): return deepcopy(value)
    frames = ('frame', 'previous_frame', 'reference_frame', 'current_frame')
    times = ('decision_ns', 'measured_ns', 'available_ns', 'reference_measured_ns', 'previous_measured_ns')
    return {k: v+1902 if k in frames and type(v) is int else
        v+1902*100_000_000 if k in times and type(v) is int else shifted(v) for k,v in value.items()}


@pytest.fixture
def intervention(monkeypatch):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    _, anchor, raw, fit, clouds, _ = transported()
    # Synthetic late-episode witness, preserving the fixed original floor reference.
    reference = deepcopy(anchor['floor_registration']['reference'])
    anchor = shifted(anchor); anchor['floor_registration']['reference'] = reference; raw = shifted(raw)
    evidence = transport_evidence(anchor, raw, fit, clouds, identity=(0, 0, 0), now_ns=191900000000,
        auxiliary_depth_sha256=raw['current_pose']['auxiliary_depth_sha256'])
    final = candidate(1904, evidence); final['original_visual_evidence'] = raw
    rows = [dict(tick=i, decision=candidate(i), input_arrays_unchanged=True,
        preintervention_complete_decision_exact=True) for i in range(1904)]
    rows[-1]['decision']['evidence'] = anchor
    rows.append(dict(tick=1904, decision=final, input_arrays_unchanged=True, preintervention_complete_decision_exact=False))
    original = dict(terminal='SENSOR_OR_MODEL_FAILURE', failure='admitted combined measured moments must reconstruct exactly',
        evidence=None, original_visual_evidence=deepcopy(raw))
    witness = dict(frame=1904, current_pose_validated=True, first_intervention=True,
        following_recorded_observations_consumed=False, candidate_decision=deepcopy(final), original_failed_decision=original)
    return json.loads(json.dumps((prefix_report(), rows, witness)))


@pytest.mark.parametrize('fault', [None, 'short', 'later', 'earlier_transport', 'summary', 'old_failure',
    'anchor', 'stale_pose', 'different_stream', 'false_input_claim', 'following'])
def test_native_admission_requires_complete_saved_current_intervention(intervention, monkeypatch, fault):
    report, rows, witness = intervention
    if fault == 'short': rows.pop()
    elif fault == 'later': rows.append(deepcopy(rows[-1]))
    elif fault == 'earlier_transport': rows[1]['decision']['evidence']['schema'] = witness['candidate_decision']['evidence']['schema']
    elif fault == 'summary': report['final_requested_command'] = [.2, 0., 0.]
    elif fault == 'old_failure': witness['original_failed_decision']['failure'] = 'different failure'
    elif fault == 'anchor': rows[1903]['decision']['evidence']['current_pose']['position_initial_body_m'][0] += .1
    elif fault == 'stale_pose': witness['candidate_decision']['evidence']['decision_ns'] -= 1
    elif fault == 'different_stream': rows[-1]['decision']['requested_command'] = [.2, 0., 0.]
    elif fault == 'false_input_claim': rows[100]['input_arrays_unchanged'] = False
    elif fault == 'following': report['following_recorded_observations_consumed'] = True
    monkeypatch.setattr(admission, 'read_json', lambda *args:witness)
    monkeypatch.setattr(admission, 'read_rows', lambda *args:iter(rows))
    if fault is None:
        r = admission.admit_intervention(None, report)
        assert r['frames'] == 1905 and r['anchor_matches_last_admitted_floor_pose']
    else:
        with pytest.raises(ValueError): admission.admit_intervention(None, report)


def function(path, name):
    return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_collector_and_audit_preserve_full_original_execution_and_evaluation():
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, n):
            n.id = {'MeasuredFloorTransportController':'DualCameraSettledController',
                'RendererWitnessDualCameraMazeSession':'DualCameraNovelMazeSession'}.get(n.id, n.id)
            return n
        def visit_Constant(self, n):
            if isinstance(n.value, str): n.value = n.value.replace('MEASURED_FLOOR_TRANSPORT_MAZE', 'DUAL_CAMERA_SETTLED_MAZE')
            return n
        def visit_Call(self, n):
            n.keywords = [k for k in n.keywords if k.arg not in
                ('measured_floor_transport_enabled', 'renderer_witnesses_recorded', 'renderer_capture_audit')]
            return self.generic_visit(n)
        def visit_Assert(self, n):
            if "result['measured_floor_transport_enabled']" in ast.unparse(n): return None
            return self.generic_visit(n)
        def visit_Assign(self, n):
            if ast.unparse(n.targets[0]) == 'renderer':
                assert ast.unparse(n.value) == 'renderer_audit(directory)'; return None
            return self.generic_visit(n)
        def visit_List(self, n):
            n.elts = [e for e in n.elts if not (isinstance(e, ast.Name) and e.id == 'RENDERER_WITNESSES')]
            return self.generic_visit(n)
    for kind, names in (('episode', ('collect','artifacts')), ('audit', ('audit',))):
        for name in names:
            old = function('scripts/dual_camera_settled_maze_'+kind+'_development.py', name)
            new = function('scripts/measured_floor_transport_maze_'+kind+'_development.py', name)
            assert ast.dump(old) == ast.dump(Normalize().visit(new)), (kind, name)


@pytest.mark.parametrize('fault', [None, 'native', 'raw', 'physics', 'public', 'context'])
def test_renderer_admission_requires_audited_same_native_startup(fault):
    r = dict(status='MAZE_RENDERER_WITNESS_INTEGRATION_COMPLETE', native_result_sha256=NATIVE_RESULT,
        actual_maze_camera_endpoints_queried=True, predecessor_outcome_unchanged=True, new_navigation_episodes=0,
        raw_sensor_reconstruction_pass=True, raw_model_command_replay_pass=True, raw_command_audit_pass=True,
        model_state_unchanged=True, comparison=dict(frames=3, physics_samples=900,
            complete_controller_decisions_exact=True, raw_captures_and_public_packets_exact=True, actual_zero_commands_exact=True,
            renderer_witnesses=dict(frames=3, capture_endpoints=6, paired_context_readbacks_equal=True, all_witnesses_match_raw_acquisitions=True)))
    if fault == 'native': r['native_result_sha256'] = '0'*64
    elif fault == 'raw': r['raw_model_command_replay_pass'] = False
    elif fault == 'physics': r['comparison']['physics_samples'] -= 1
    elif fault == 'public': r['comparison']['raw_captures_and_public_packets_exact'] = False
    elif fault == 'context': r['comparison']['renderer_witnesses']['all_witnesses_match_raw_acquisitions'] = False
    if fault is None: admit_renderer(r)
    else:
        with pytest.raises(ValueError): admit_renderer(r)
