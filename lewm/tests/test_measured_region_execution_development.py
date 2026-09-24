import ast
import inspect
import textwrap

from scripts import run_go2_measured_region_navigation_development_v1 as runner
from scripts import run_go2_moving_rgbd_whole_task_development_v1 as prior
from scripts import audit_measured_region_rgb_core_development as core
from scripts import audit_go2_whole_task_navigation_sampling_correction_development_v1 as old_core
from lewm.measured_region_navigation_development import MeasuredRegionNavigation


def test_distinct_controller_schema_and_same_two_physical_constructions():
    assert runner.SCHEMA=='measured_region_navigation_development.v1'
    assert runner.WholeTaskNavigation is MeasuredRegionNavigation
    for a,b in zip(runner.trial_specs(),prior.trial_specs(),strict=True):
        assert a['method']=='measured_region' and a['scene_id']!=b['scene_id']
        assert {k:v for k,v in a.items() if k not in ('scene_id','method')}=={k:v for k,v in b.items() if k not in ('scene_id','method')}


def test_collect_changes_only_explicit_controller_observation_inputs():
    source=inspect.getsource(prior.collect).replace(
        "decision = controller.observe(packet, session.fast_buffer.packet(now_ns=now), now_ns=now)",
        "decision = controller.observe_rgbd(packet, session.fast_buffer.packet(now_ns=now), session.latest_depth,\n                                                       session.relative_observations[index]['observer'], now_ns=now)")
    assert ast.dump(ast.parse(textwrap.dedent(source)))==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(runner.collect))))


def test_full_physical_audit_retained_with_only_explicit_new_controller_replay():
    source=inspect.getsource(old_core.audit_trial)
    source=source.replace("decisions = json.loads((directory / 'task_decisions.json').read_text())",
        "decisions = json.loads((directory / 'task_decisions.json').read_text())\n    relative_records = json.loads((directory/'relative_state_observations.json').read_text())")
    source=source.replace("expected = controller.observe(packet, fast, now_ns=decision['decision_ns'])",
        "expected = controller.observe_rgbd(packet, fast, load_rgbd_observation(directory, decision['observation_index'])[1],\n            relative_records[decision['observation_index']]['observer'], now_ns=decision['decision_ns'])")
    source=source.replace("controller.observe(packet, load_fast_packet(directory, fault['observation_index']), now_ns=packet['sensor_state']['decision_ns'])",
        "controller.observe_rgbd(packet, load_fast_packet(directory, fault['observation_index']),\n                load_rgbd_observation(directory, fault['observation_index'])[1], relative_records[fault['observation_index']]['observer'],\n                now_ns=packet['sensor_state']['decision_ns'])")
    assert ast.dump(ast.parse(textwrap.dedent(source)))==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(core.audit_trial))))
