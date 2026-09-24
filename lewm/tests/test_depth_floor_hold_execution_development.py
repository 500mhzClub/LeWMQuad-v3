import ast
import inspect
import textwrap

from scripts import run_go2_depth_floor_hold_navigation_development_v1 as runner
from scripts import run_go2_observable_hold_navigation_development_v1 as prior
from scripts import audit_depth_floor_hold_rgb_core_development as core
from scripts import audit_observable_hold_rgb_core_development as old_core
from lewm.depth_floor_hold_navigation_development import DepthFloorHoldNavigation


def test_explicit_new_controller_and_same_full_missions():
    assert runner.SCHEMA=='depth_floor_hold_navigation_development.v1'
    assert runner.WholeTaskNavigation is core.WholeTaskNavigation is DepthFloorHoldNavigation
    assert runner.OUTPUT!=prior.OUTPUT
    for a,b in zip(runner.trial_specs(),prior.trial_specs(),strict=True):
        assert a['method']=='depth_floor_hold' and a['scene_id']!=b['scene_id']
        assert {k:v for k,v in a.items() if k not in ('scene_id','method')}=={k:v for k,v in b.items() if k not in ('scene_id','method')}


def test_full_collection_and_physical_audit_bodies_unchanged():
    for a,b in [(runner.collect,prior.collect),(core.audit_trial,old_core.audit_trial)]:
        assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(a))))==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(b))))
