"""Full synthetic replay and original corruption checks for the new composition."""
from itertools import islice
from types import FunctionType

from scripts import single_pass_body_projected_replay_development as run
from lewm import single_pass_body_projected_controller_development as candidate
from lewm import body_projected_tiled_controller_development as body
from lewm.tests import test_body_projected_tiled_replay_development as previous


def fixture(monkeypatch, tmp_path, fault=None):
    rows, prior, tape, calls, models = previous.fixture(monkeypatch, tmp_path, fault)
    fake = previous.run.BodyProjectedTiledController
    class Baseline(fake):
        index = 0
    class Candidate(Baseline):
        index = 1
        def observe(self, *args, **kwargs):
            return super().observe(*args, **kwargs) | {'controller': candidate.CONTROLLER, candidate.FLAG: True}
    monkeypatch.setattr(run, 'BodyProjectedTiledController', Baseline)
    monkeypatch.setattr(run, 'SinglePassBodyProjectedController', Candidate)
    monkeypatch.setattr(run, 'OUTPUT', tmp_path)
    monkeypatch.setattr(run, 'normalized_state_tree', run.original.state_tree)
    for i, row in enumerate(islice(run.original.profile.read_rows(None), 1428)):
        decision = row['decision'] | {run.original.FLAG: True, 'controller': body.CONTROLLER,
            body.FLAG: True, previous.visible.FLAG: True, previous.progressive.FLAG: True,
            previous.density.FLAG: True, previous.visibility.FLAG: True,
            previous.previous.fused.FLAG: True, previous.previous.combined.FLAG: True,
            previous.packed_base.FLAG: True, previous.packed.FLAG: True,
            'batched_retained_floor_queries_enabled': True}
        rows[i]['candidate_decision_sha256'] = run.original.profile.reference.saved.identity(decision)
    return rows, prior, tape, calls, models


for name in ('test_original_corruption_rejections', 'test_reference_binding_rejections'):
    function = getattr(previous, name)
    clone = FunctionType(function.__code__, function.__globals__ | dict(run=run, fixture=fixture),
        name, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__dict__.update(function.__dict__)
    globals()[name] = clone


def test_original_loop_private_bindings_and_unchanged_imported_globals():
    old = run.original.replay
    before = old.__globals__.copy()
    revised = run.isolated_replay()
    assert revised.__code__ is old.__code__ and revised.__closure__ is old.__closure__
    assert revised.__globals__['FrozenFootprintAnchoredController'] is run.BodyProjectedTiledController
    assert revised.__globals__['ScopedFootprintAnchoredController'] is run.SinglePassBodyProjectedController
    assert revised.__globals__['profile'].normalize_candidate is run.previous.normalize_candidate
    assert all(old.__globals__[k] is v for k, v in before.items())


def test_all_1428_observations_independent_models_and_seven_states(monkeypatch, tmp_path):
    rows, prior, _, calls, models = fixture(monkeypatch, tmp_path)
    report = run.replay(rows, prior)
    assert calls == [(i, j) for i in range(1428) for j in run.original.execution_order(i)]
    assert models[0] is not models[1]
    assert report['frames'] == 1428 and report['raw_model_forecast_comparisons'] == 1425
    assert report['observed_state_checks'] == prior['observed_state_checks']
    assert report['baseline'] == 'BodyProjectedTiledController'
    assert report['candidate'] == 'SinglePassBodyProjectedController'
    assert report['original_packed_insertion_unchanged']
