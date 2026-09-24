"""Synthetic end-to-end wiring, retained state and changed pending forecasts."""
import ast
from copy import deepcopy
import inspect
from types import SimpleNamespace
import pytest
from lewm.tests.test_commitment_contact_anchored_development import pair
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from scripts import replay_go2_commitment_contact_anchored_prefix_v1 as runner
from scripts import replay_go2_hold_reorientation_maze02_prefix_v1 as reference


def test_original_worker_admission_changes_only_fixed_case_index_and_message():
    old = inspect.getsource(reference.admit_worker)
    wanted = old.replace('original.CASES[0]', 'original.CASES[1]').replace(
        'unchanged exact original first adapter case required', 'unchanged exact original second adapter case required')
    assert ast.dump(ast.parse(inspect.getsource(runner.admit_worker))) == ast.dump(ast.parse(wanted))


def fixture(monkeypatch, tmp_path, fault=None):
    pairs = []
    for frame in range(4):
        a,b = pair()
        for d in (a,b):
            d['tick'] = frame
            if frame < 3: d.update(new_selection=None, requested_command=[0., 0., 0.], selected_action=None)
        pairs.append((a,b))
    rows = [dict(tick=i, decision=a) for i,(a,b) in enumerate(pairs)]
    expected = [dict(frame=i, original_row_sha256=runner.saved.identity(row),
        candidate_selection_sha256=runner.saved.identity(pairs[i][1]['new_selection'])) for i,row in enumerate(rows)]
    tape = [dict(tick=i, completed=True, pre_sample_index=749+50*i, post_sample_index=799+50*i,
        requested_command=pairs[i][0]['requested_command']) for i in range(4)]
    boundary = dict(original_requested_command=pairs[-1][0]['requested_command'],
        candidate_requested_command=pairs[-1][1]['requested_command'])
    if fault == 'incomplete': tape[-1]['completed'] = False
    if fault == 'clock': tape[-1]['post_sample_index'] += 1
    if fault == 'saved_row': expected[1]['original_row_sha256'] = 'changed'
    seen = []
    def stream(directory):
        yield from deepcopy(rows)
        raise AssertionError('post-intervention original observation consumed')
    class Reader:
        def __init__(self, directory): pass
        def packet(self, frame):
            assert frame < 4
            seen.append(frame)
            return {'frame':frame}, {}, {}, 1_500_000_000+frame*100_000_000
    class Controller:
        def __init__(self, *args, candidate=False, **kwargs):
            self.candidate = candidate; self.memory = {'witness':['same']}; self.history = []
            self.mapper = SimpleNamespace(floor={(0,0)}, occupied=set())
            self.residual = SimpleNamespace(pending=None, pose=None, frame=-1, now_ns=None, history=[])
        def observe(self, p, *args, now_ns, **kwargs):
            frame = p['frame']; self.history.append(deepcopy(p))
            result = deepcopy(pairs[frame][int(self.candidate)])
            self.residual.frame = frame; self.residual.now_ns = now_ns
            self.residual.pose = dict(position=[0.,0.,0.], rotation=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                rgb_sha256='rgb', depth_sha256='depth')
            selection = result['new_selection']; self.residual.pending = None
            if selection:
                i = next(i for i,a in enumerate(ACTIONS) if result['requested_command'] == candidate_commands(a)[0])
                self.residual.pending = dict(tick=frame, measured_ns=now_ns, action=ACTIONS[i],
                    requested_command=result['requested_command'], predicted_body_xy_m=selection['prediction'][i][0][:2],
                    **deepcopy(self.residual.pose))
            if frame == 3:
                if fault == 'raw_original' and not self.candidate: result['evidence']['observed'] = -1
                if fault == 'original_mutation' and not self.candidate: p['changed'] = True
                if self.candidate:
                    if fault == 'contact': self.memory['changed'] = True
                    if fault == 'map': self.mapper.floor.add((1,1))
                    if fault == 'mutation': p['changed'] = True
                    if fault == 'pending': self.residual.pending['predicted_body_xy_m'] = [99.,99.]
                    if fault == 'pending_pose': self.residual.pending['position'] = [99.,0.,0.]
                    if fault == 'observed_residual': self.residual.history.append({'changed':True})
                    if fault == 'history': self.history.append({'changed':True})
                    if fault == 'candidate_selection': result['new_selection']['utility_fault'] = True
            return result
    model = SimpleNamespace(state_dict=lambda: {}, parameters=lambda: [])
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setattr(runner, 'saved_inputs', lambda: ({}, deepcopy(boundary), deepcopy(expected)))
    monkeypatch.setattr(runner, 'read_rows', stream)
    monkeypatch.setattr(runner, 'IntentReturnRGBDReplay', Reader)
    monkeypatch.setattr(runner, 'packet', lambda *a, **k: ({}, {}))
    monkeypatch.setattr(runner, 'public_acquisition', lambda x: x)
    monkeypatch.setattr(runner, 'read_json', lambda root,name: deepcopy(tape) if name == 'command_tape.json'
        else [{}]*4 if name == 'auxiliary_camera_audit.json' else {})
    monkeypatch.setattr(runner, 'ArticulatedCollisionGeometry', lambda x: object())
    monkeypatch.setattr(runner, 'ResidualAnchoredContinuationController', Controller)
    monkeypatch.setattr(runner, 'CommitmentContactAnchoredController', lambda *a, **k: Controller(*a, candidate=True, **k))
    monkeypatch.setattr(runner.original, 'assigned_model', lambda *a: model if fault == 'shared_model'
        else SimpleNamespace(state_dict=lambda: {}, parameters=lambda: []))
    monkeypatch.setattr(runner, 'state_digest', lambda x: runner.MODEL_SHA)
    return seen


def test_complete_synthetic_replay_checks_changed_forecast_and_stops_at_four(monkeypatch, tmp_path):
    seen = fixture(monkeypatch, tmp_path)
    result = runner.replay()
    assert seen == [0,1,2,3] and result['frames'] == 4
    assert result['raw_model_forecast_comparisons'] == 1 and result['first_changed_command_frame'] == 3
    assert result['complete_retained_observed_state_exact'] and result['selected_pending_forecasts_checked']
    assert not result['native_execution']


@pytest.mark.parametrize('fault', ['incomplete','clock','saved_row','raw_original','contact','map',
    'mutation','original_mutation','pending','pending_pose','observed_residual','history','candidate_selection','shared_model'])
def test_input_state_and_selected_forecast_faults_stop_raw_pipeline(monkeypatch, tmp_path, fault):
    fixture(monkeypatch, tmp_path, fault)
    with pytest.raises(ValueError): runner.replay()


@pytest.mark.parametrize('ram,disk', [(47,100), (64,40)])
def test_resource_shortfall_prevents_replay(ram,disk):
    with pytest.raises(ValueError): runner.resources_for(dict(memory_available_bytes=ram*1024**3,
        artifact_free_bytes=disk*1024**3))
