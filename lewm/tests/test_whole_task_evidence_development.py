"""Collector/auditor wiring, physical object integrity and matched pre-discovery actions."""
import ast
from copy import deepcopy
import inspect
import math

import numpy as np
import pytest

from lewm.tests.test_rgb_marker_beacon_development import static_rows
from lewm.tests.test_whole_task_navigation_development import controller, marker_pixels
from lewm.tests.test_observed_continuation_development import MovingStream
from lewm.whole_task_navigation_development import WholeTaskNavigation, MAX_SECONDS, MAX_LEGS
from lewm.whole_task_rgb_dataset_development import load_whole_task_observation
from lewm.whole_task_scene_development import trial_specs
from scripts import run_go2_whole_task_navigation_development_v1 as runner
from scripts import audit_go2_whole_task_navigation_development_v1 as auditor


def native_rows(spec):
    rows = deepcopy(static_rows(spec))
    for box, row in zip(spec['geometry']['wall_boxes'], rows, strict=True):
        # The source fixture aliases pack dimensions and native dimensions;
        # real native evidence has a separate padded array, not a seven-D pack.
        row['native_box_size'] = [*row['native_box_size'], 0., 0., 0., 0.]
        row['native_position'] = np.asarray(box['centre_xyz'], dtype=np.float32).astype(float).tolist()
        row['pack_object']['yaw_rad'] = box['yaw_rad']
        row['native_quaternion_wxyz'] = [math.cos(box['yaw_rad']/2), 0., 0., math.sin(box['yaw_rad']/2)]
    return rows


def test_all_rotated_marker_and_wall_collision_identities_are_checked():
    for spec in trial_specs(): auditor.check_static_objects(spec, native_rows(spec))


@pytest.mark.parametrize('fault', ['padding', 'rotation', 'color', 'missing', 'position'])
def test_changed_physical_marker_or_environment_cannot_pass_audit(fault):
    spec = trial_specs()[2]; rows = native_rows(spec)
    if fault == 'padding': rows[-1]['native_box_size'][-1] = 1.
    if fault == 'rotation': rows[-1]['native_quaternion_wxyz'] = [1., 0., 0., 0.]
    if fault == 'color': rows[-1]['surface_rgb'] = [.35, .35, .35]
    if fault == 'missing': rows.pop(0)
    if fault == 'position': rows[-1]['native_position'][0] += .001
    with pytest.raises(ValueError): auditor.check_static_objects(spec, rows)


def test_collector_auditor_share_runtime_and_long_reader_with_no_map_input():
    assert runner.WholeTaskNavigation is auditor.WholeTaskNavigation is WholeTaskNavigation
    assert auditor.load_route_observation is load_whole_task_observation
    assert MAX_SECONDS == 360 and MAX_LEGS == 36
    for function in (runner.collect, auditor.audit_trial):
        tree = ast.parse(inspect.getsource(function))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'WholeTaskNavigation']
        assert len(calls) == 1
        call = calls[0]
        assert len(call.args) == 2 and ast.unparse(call.args[0]) == "spec['method']"
        assert [(k.arg, ast.unparse(k.value)) for k in call.keywords] == [('memory_arm', "spec['memory_arm']")]
    assert len(trial_specs()) == 4 and {s['method'] for s in trial_specs()} == {'fixed_forward'}


def test_memory_ablation_has_identical_pre_discovery_decisions_and_first_return_selection():
    remembered, local = controller('episodic'), controller('local_only')
    stream = MovingStream(); command = [0., 0., 0.]; activated = False
    found = False
    for tick in range(1200):
        child = remembered.child if remembered.stage == 'TRAVERSE' else None
        p, fast, now = stream.frame(tick, command, changed=child is not None and child.tick >= 3)
        if tick == 0: p['image']['rgb'][:120] = [180, 180, 180]
        if remembered.completed_legs >= 2 and remembered.stage == 'SCAN': activated = True
        if activated: marker_pixels(p)
        a = remembered.observe(p, fast, now_ns=now)
        b = local.observe(p, fast, now_ns=now)
        omit = ('memory_arm', 'memory_digest')
        assert {k: v for k, v in a.items() if k not in omit} == {k: v for k, v in b.items() if k not in omit}
        if a['mission'] == 'RETURN' and a['selected_branch'] is not None:
            found = True; break
        assert not a['terminal']
        command = a['requested_command']
    assert found and remembered.memory is not None and local.memory is None
