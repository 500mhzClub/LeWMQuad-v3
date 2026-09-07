import ast
import inspect
import textwrap

from scripts import run_go2_observable_hold_navigation_development_v1 as runner
from scripts import run_go2_release_aware_navigation_development_v1 as prior
from scripts import audit_observable_hold_rgb_core_development as core
from scripts import audit_release_aware_rgb_core_development as old_core
from lewm.observable_hold_navigation_development import ObservableHoldNavigation


def test_explicit_new_controller_and_same_two_physical_missions():
    assert runner.SCHEMA=='observable_hold_navigation_development.v1'
    assert runner.WholeTaskNavigation is core.WholeTaskNavigation is ObservableHoldNavigation
    assert runner.OUTPUT!=prior.OUTPUT
    for a,b in zip(runner.trial_specs(),prior.trial_specs(),strict=True):
        assert a['method']=='observable_hold' and a['scene_id']!=b['scene_id']
        assert {k:v for k,v in a.items() if k not in ('scene_id','method')}=={k:v for k,v in b.items() if k not in ('scene_id','method')}


def test_collection_and_complete_physical_audit_bodies_are_unchanged():
    for a,b in [(runner.collect,prior.collect),(core.audit_trial,old_core.audit_trial)]:
        assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(a))))==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(b))))
