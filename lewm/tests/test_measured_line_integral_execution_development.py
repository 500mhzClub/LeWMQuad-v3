import ast
import inspect
import textwrap

from scripts import run_go2_measured_line_integral_navigation_development_v1 as runner
from scripts import run_go2_measured_region_navigation_development_v1 as prior
from scripts import audit_measured_line_integral_rgb_core_development as core
from scripts import audit_measured_region_rgb_core_development as old_core
from lewm.measured_line_integral_navigation_development import MeasuredLineIntegralNavigation


def test_distinct_controller_and_same_two_full_missions():
    assert runner.SCHEMA=='measured_line_integral_navigation_development.v1'
    assert runner.WholeTaskNavigation is core.WholeTaskNavigation is MeasuredLineIntegralNavigation
    assert runner.OUTPUT != prior.OUTPUT
    for a,b in zip(runner.trial_specs(),prior.trial_specs(),strict=True):
        assert a['method']=='measured_line_integral' and a['scene_id']!=b['scene_id']
        assert {k:v for k,v in a.items() if k not in ('scene_id','method')}=={k:v for k,v in b.items() if k not in ('scene_id','method')}


def test_collection_and_full_core_audit_bodies_are_unchanged():
    for a,b in [(runner.collect,prior.collect),(core.audit_trial,old_core.audit_trial)]:
        assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(a))))==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(b))))
