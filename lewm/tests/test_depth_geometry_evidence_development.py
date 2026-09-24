import ast
import inspect
import textwrap

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.depth_geometry_evaluation_development import expected_optical_depth, evaluate_depth
from scripts import run_go2_marker_beacon_development_v1 as original
from scripts import run_go2_rgbd_observation_development_v1 as runner
from scripts.rgbd_session_development import RGBDSession
from scripts.whole_task_physics_session_development import WholeTaskPhysicsSession


def test_analytic_reference_has_visible_and_occluded_surfaces_and_background():
    transform = np.array(BODY_FROM_OPTICAL); transform[2, 3] += .32
    for spec in runner.trials():
        ref = expected_optical_depth(spec['geometry']['wall_boxes'], transform)
        native = np.full((480, 640), 200., np.float32)
        native[np.ix_(ref['rows'], ref['columns'])] = np.where(np.isfinite(ref['expected_depth_m']), ref['expected_depth_m'], 200.)
        result = evaluate_depth(native, spec['geometry']['wall_boxes'], transform, marker_case=spec['marker_case'])
        assert result['passes_declared_depth_check']
        assert (result['background_rays'] > 0) == (spec['marker_case'] == 'positive')
        u, v = np.meshgrid(ref['columns']+.5, ref['rows']+.5)
        native[np.ix_(ref['rows'], ref['columns'])] *= np.sqrt(1+((u-320)/FOCAL)**2+((v-240)/FOCAL)**2)
        wrong = evaluate_depth(native, spec['geometry']['wall_boxes'], transform, marker_case=spec['marker_case'])
        assert not wrong['checks']['metric_optical_depth_within_5mm']


def test_raw_collector_body_reused_without_changing_gait_or_marker_detector():
    def syntax(function): return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(function))))
    assert syntax(runner.collect) == syntax(original.collect)
    assert runner.RouteSession is RGBDSession and issubclass(RGBDSession, WholeTaskPhysicsSession)
    assert RGBDSession.command_tick is WholeTaskPhysicsSession.command_tick
    assert RGBDSession.settle_recorded is WholeTaskPhysicsSession.settle_recorded
    assert RGBDSession._sample is WholeTaskPhysicsSession._sample
    assert len(runner.trials()) == 2 and all(set(s['geometry']) == {'spawn_se2_world', 'wall_boxes'} for s in runner.trials())


def test_native_acquisition_requests_rgb_and_depth_together_not_geometry_labels():
    tree = ast.parse(textwrap.dedent(inspect.getsource(RGBDSession.capture_fixed_rgb)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == 'render']
    assert len(calls) == 1
    assert {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords} == dict(rgb=True, depth=True, segmentation=False, normal=False)
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'geometry' for n in ast.walk(tree))
    assert len(runner.verify_native()) == 5
