"""Comparator audit attribution and guarded execution without a native scene."""
import inspect
import json
from types import SimpleNamespace as NS

import pytest
import torch

from scripts import extended_return_budget_comparator_pipeline_development as pipeline


@pytest.mark.parametrize('mode', pipeline.MODES)
def test_original_physics_and_complete_audit_code_and_larger_dependencies(mode):
    collect, audit = pipeline.functions(mode)
    old_collect, old_audit = ((pipeline.reactive_episode.collect, pipeline.reactive_audit.audit)
        if mode == 'reactive' else (pipeline.extended.collect, pipeline.extended.audit))
    for actual, original in ((collect, old_collect), (audit, old_audit)):
        assert actual.__code__ is original.__code__ and actual.__closure__ is original.__closure__
        assert actual.__defaults__ is original.__defaults__ and actual.__kwdefaults__ == original.__kwdefaults__
        assert actual.__globals__['NAVIGATION_TICKS'] == 8000
        assert actual.__globals__['specification'] is original.__globals__['specification']
        assert actual.__globals__['public_mission'] is original.__globals__['public_mission']
    assert collect.__globals__['MAX_OBSERVATIONS'] == 8014
    assert collect.__globals__['COLLECTION_ALLOWANCE_BYTES'] == 28*1024**3
    assert collect.__globals__['writer'] is pipeline.extended.writer
    assert audit.__globals__['MAX_COMMAND_TICKS'] == 8013
    for key, expected in dict(IntentReturnRGBDReplay=pipeline.extended.ExtendedReturnBudgetRGBDReplay,
            audit_sensors=pipeline.extended.audit_sensors, read_rows=pipeline.extended.read_rows,
            packet=pipeline.extended.rgb_packet, renderer_audit=pipeline.extended.renderer_audit,
            evaluate=pipeline.extended.evaluate).items():
        assert audit.__globals__[key] is expected
    assert audit.__globals__['audit_commands'].__globals__['NAVIGATION_TICKS'] == 8000
    if mode == 'reactive': assert 'model' not in inspect.signature(collect).parameters
    definition = pipeline.definition(mode)
    assert definition['sampled_resource_guards_enabled'] and definition['navigation_ticks'] == 8000
    assert not definition['population_execution_permitted'] and not definition['single_read_auxiliary_acquisition_adopted']
    if mode == 'reactive':
        assert definition['fully_nonpredictive_controller'] and definition['reactive_is_whole_method_comparison']
        assert not definition['high_level_world_model_loaded'] and not definition['predictive_surface_or_path_gates_applied']
    elif mode == 'nominal':
        assert definition['nominal_predictive_controller'] and not definition['learned_model_forward_permitted']
        assert not definition['fully_nonpredictive_controller']
    elif mode == 'current_planning':
        assert not definition['memoryless_controller'] and not definition['accumulated_planning_cells_queried']


# Explicit globals allow the real executor to perform its checked private bind.
ResidualAnchoredContinuationController = ReactiveFloorTransportController = None
RendererWitnessDualCameraMazeSession = audit_sensors = None


def synthetic_collect(*args, output, episode_name, **kwargs):
    session = RendererWitnessDualCameraMazeSession()
    constructor = ReactiveFloorTransportController if 'model' not in kwargs else ResidualAnchoredContinuationController
    controller = constructor(**kwargs)
    try:
        for frame in range(3):
            session.samples = range(750+50*frame)
            packet = session.sensor_packets(); controller.observe(packet)
        return {'synthetic_collection': True}
    finally: (output/(episode_name+'_persisted')).write_text('original cleanup')


def synthetic_audit(*args, **kwargs):
    packet = audit_sensors()
    reactive = 'model' not in kwargs
    constructor = ReactiveFloorTransportController if reactive else ResidualAnchoredContinuationController
    constructor(**kwargs).observe(packet)
    flag = 'raw_controller_command_replay_pass' if reactive else 'raw_model_command_replay_pass'
    return {flag: True, 'raw_sensor_reconstruction_pass': True, 'raw_command_audit_pass': True,
        'model_state_unchanged': True}


def setup(tmp_path, monkeypatch, mode, fault=None):
    calls = []; state = dict(monotonic_s=0., rss_bytes=1024**3,
        memory_available_bytes=70*1024**3, artifact_free_bytes=100*1024**3)
    monkeypatch.setattr(pipeline.resources, 'validate_root', lambda root: root)
    def snapshot(root): state['monotonic_s'] += 1.; return dict(state)
    monkeypatch.setattr(pipeline.resources, 'snapshot', snapshot)
    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).eval()
    class Session:
        def sensor_packets(self): return {'frame': (len(self.samples)-750)//50}
    class Controller:
        def __init__(self, **kwargs):
            calls.append('constructor')
        def observe(self, packet):
            calls.append(packet)
            if fault in ('root_forward', 'child_forward', 'swallowed_forward'):
                target = model[0] if fault == 'child_forward' else model
                try: target(torch.ones(1, 1))
                except RuntimeError:
                    if fault != 'swallowed_forward': raise
            if fault == 'after_controller': state['rss_bytes'] = 49*1024**3
            if fault == 'original_failure': raise KeyError('original controller error')
            return {'same_decision': packet}
    monkeypatch.setitem(pipeline.CONTROLLERS, mode, Controller)
    monkeypatch.setattr(pipeline.guarded.pipeline, 'ExtendedReturnBudgetRendererSession', Session)
    monkeypatch.setattr(pipeline, 'functions', lambda selected: (synthetic_collect, synthetic_audit))
    monkeypatch.setattr(pipeline.extended, 'audit_sensors', lambda: {'raw': 'same'})
    kwargs = dict(episode_name='synthetic')
    if mode != 'reactive': kwargs.update(model=model, condition='direct', variant='no_rgb')
    return model, calls, kwargs


@pytest.mark.parametrize('mode', pipeline.MODES)
def test_guarded_collection_and_audit_keep_decisions_and_actual_scope(tmp_path, monkeypatch, mode):
    model, calls, kwargs = setup(tmp_path, monkeypatch, mode)
    collection = pipeline.collect(mode=mode, output=tmp_path, **kwargs)
    assert collection == {'synthetic_collection': True}
    assert (tmp_path/'synthetic_persisted').is_file()
    audit = pipeline.audit(mode=mode, input_root=tmp_path, **kwargs)
    assert audit['raw_sensor_reconstruction_pass'] and audit['raw_command_audit_pass']
    assert calls == ['constructor', {'frame': 0}, {'frame': 1}, {'frame': 2}, 'constructor', {'raw': 'same'}]
    if mode == 'nominal':
        assert not audit['raw_model_command_replay_pass'] and not audit['high_level_world_model_used']
        assert audit['actual_learned_model_forward_calls'] == 0
        assert audit['raw_nominal_forecast_command_replay_pass'] and audit['raw_controller_command_replay_pass']
        assert not audit['fully_nonpredictive_controller']
    for phase in ('collection', 'audit'):
        report = json.loads((tmp_path/pipeline.resources.names('synthetic', phase)[1]).read_text())
        assert report['phase_completed'] and report['sampled_limits_passed']
    assert all(not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize('phase', ['collection', 'audit'])
@pytest.mark.parametrize('fault', ['root_forward', 'child_forward', 'swallowed_forward'])
def test_nominal_inference_attempt_cannot_be_hidden_and_fails_resource_completion(tmp_path, monkeypatch, phase, fault):
    model, _, kwargs = setup(tmp_path, monkeypatch, 'nominal', fault)
    function = pipeline.collect if phase == 'collection' else pipeline.audit
    root_key = 'output' if phase == 'collection' else 'input_root'
    with pytest.raises((RuntimeError, ValueError), match='model'):
        function(mode='nominal', **{root_key:tmp_path}, **kwargs)
    report = json.loads((tmp_path/pipeline.resources.names('synthetic', phase)[1]).read_text())
    assert not report['phase_completed'] and report['error']
    assert all(not module._forward_pre_hooks for module in model.modules())
    if phase == 'collection': assert (tmp_path/'synthetic_persisted').is_file()


@pytest.mark.parametrize('mode', pipeline.MODES)
@pytest.mark.parametrize('fault', ['after_controller', 'original_failure'])
def test_breach_or_original_error_preserves_cleanup_and_stops_next_call(tmp_path, monkeypatch, mode, fault):
    _, calls, kwargs = setup(tmp_path, monkeypatch, mode, fault)
    with pytest.raises((pipeline.resources.ResourceLimitError, KeyError)):
        pipeline.collect(mode=mode, output=tmp_path, **kwargs)
    assert calls == ['constructor', {'frame': 0}]
    assert (tmp_path/'synthetic_persisted').is_file()
    report = json.loads((tmp_path/pipeline.resources.names('synthetic', 'collection')[1]).read_text())
    assert not report['phase_completed'] and report['error']


@pytest.mark.parametrize('mode', [None, 'learned', True])
def test_invalid_mode_never_creates_resource_or_episode_files(tmp_path, mode):
    with pytest.raises(ValueError, match='explicit prepared'):
        pipeline.collect(mode=mode, output=tmp_path, episode_name='synthetic')
    assert not list(tmp_path.iterdir())


def test_wrong_model_scope_is_rejected_before_resources(tmp_path):
    with pytest.raises(ValueError, match='no model'):
        pipeline.collect(mode='reactive', output=tmp_path, episode_name='synthetic', model=None)
    with pytest.raises(ValueError, match='caller-admitted'):
        pipeline.collect(mode='nominal', output=tmp_path, episode_name='synthetic', model=None)
    assert not list(tmp_path.iterdir())
