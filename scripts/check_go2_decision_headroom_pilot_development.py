"""Small synthetic checks for the audit, without native physics or model calls."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from lewm.decision_headroom_packet_development import check_retained_inputs
from lewm.decision_headroom_reference_development import ReferenceGeometry, fixed_target_world
from scripts.run_go2_decision_headroom_branches_development import compare_trace, swept_clearance
from scripts.run_go2_decision_headroom_pilot_development import CAPS, PilotBudget


def trace():
    times = np.arange(401)*.002
    pose = np.zeros((401, 7)); pose[:, 6] = 1
    pose[:, 0] = times*.1
    return dict(timestamp_s=times, base_pose_world=pose, physics_contact=np.zeros(401, np.uint8))


class AuditChecks(unittest.TestCase):
    def test_recorded_target_is_not_reanchored_at_current_pose(self):
        c, s = np.cos(np.pi/4), np.sin(np.pi/4)
        B = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        packet = dict(active_target=dict(kind='observed_map_xy',xy=[2.,0.]),
            observed_map=SimpleNamespace(map_from_initial=B), observed_position=np.array([0.,0.,.1]))
        initial = np.array([10.,10.,.3,0.,0.,np.sin(np.pi/4),np.cos(np.pi/4)])
        result = fixed_target_world(packet, initial)
        np.testing.assert_allclose(result['target_xy_world'],[10+np.sqrt(2),10+np.sqrt(2)])
        packet['observed_position'][:2] = [50.,-60.]
        self.assertEqual(result, fixed_target_world(packet, initial))
        packet['active_target'] = dict(kind='NO_XY_ROUTE_TARGET')
        self.assertFalse(fixed_target_world(packet, initial)['valid'])

    def test_systematic_restore_error_is_not_repeat_noise(self):
        original = trace()
        restored = {k:v.copy() for k,v in original.items()}
        restored['base_pose_world'][:, 0] += .002
        tolerance = dict(position_m=.001, yaw_rad=np.deg2rad(.1))
        self.assertTrue(compare_trace(restored, restored, 0, tolerance)['passed'])
        result = compare_trace(restored, original, 0, tolerance)
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['horizons']), 9)
        self.assertTrue(all(not r['passed'] for r in result['horizons']))

    def test_intermediate_contact_and_missing_horizon_are_not_hidden(self):
        original = trace()
        changed = {k:v.copy() for k,v in original.items()}
        changed['physics_contact'][37] = 1
        tolerance = dict(position_m=.001, yaw_rad=.01)
        result = compare_trace(changed, original, 0, tolerance)
        self.assertFalse(result['passed'])
        self.assertFalse(result['horizons'][1]['contact_match'])
        incomplete = {k:v[:-1] for k,v in original.items()}
        result = compare_trace(incomplete, original, 0, tolerance)
        self.assertFalse(result['passed'])
        self.assertEqual(result['horizons'][-1]['reason'], 'MISSING_OR_DUPLICATE_PHYSICAL_SAMPLE')

    def test_sweep_bound_catches_between_sample_crossing(self):
        geometry = ReferenceGeometry([dict(center=[0,0], size=[.02,2], yaw=0)],
            [[-3,-3],[3,3]], [1,0], radius_m=.1, clearance_m=.005, resolution_m=.02)
        arrays = trace()
        arrays = {k:v[:2].copy() for k,v in arrays.items()}
        arrays['base_pose_world'][:, :2] = [[-.2,0],[.2,0]]
        result = swept_clearance(arrays, geometry, 0)
        self.assertGreater(result['minimum_sampled_clearance_m'], .005)
        self.assertLess(result['minimum_interpolated_clearance_lower_bound_m'], 0)
        self.assertFalse(geometry.segment_clear([-.2,0],[.2,0]))

    def test_retention_rejects_prohibited_nested_arrays(self):
        with self.assertRaisesRegex(ValueError, 'raw depth'):
            check_retained_inputs(dict(evidence=SimpleNamespace(depth_m=np.zeros((8,8)))))
        with self.assertRaisesRegex(ValueError, 'dense feature'):
            check_retained_inputs(dict(features=np.zeros((1,768,1024))))
        check_retained_inputs(dict(depth=dict(sha256='recorded digest'), rgb=np.zeros((480,640,3), np.uint8)))

    def test_budget_population_cannot_be_restarted_or_expanded(self):
        # No filesystem, GPU measurement or owner is needed to exercise the
        # actual reservation rules. This cannot execute source or branch work.
        budget = PilotBudget.__new__(PilotBudget)
        budget.caps = json.loads(CAPS.read_text())
        budget.sources, budget.snapshots, budget.branches = set(), set(), set()
        budget.components = set()
        budget.source_ns, budget.active_case = {}, None
        budget.check = lambda *args, **kwargs: None
        budget.event = lambda *args, **kwargs: None
        for case in range(6):
            budget.start_source(case)
            budget.reserve_source_physics(case, 1.5)
            for _ in range(805):
                budget.reserve_source_physics(case, .02)
            with self.assertRaisesRegex(RuntimeError, 'source physics cap'):
                budget.reserve_source_physics(case, .02)
            for frame in (12,52,92,132):
                budget.reserve_snapshot(case, frame)
                prefix = f'source_{case:02d}/state_{frame:04d}/'
                for repeat in range(3):
                    budget.reserve_branch(prefix+f'source_trace_{repeat}')
                    for action in ('hold','forward','left_arc','right_arc','left_turn','right_turn'):
                        budget.reserve_branch(prefix+f'{action}/{repeat}')
                with self.assertRaises(ValueError):
                    budget.reserve_branch(prefix+'hold/0')
                with self.assertRaises(ValueError):
                    budget.reserve_branch(prefix+'new_action/0')
                with self.assertRaises(ValueError):
                    budget.reserve_branch(prefix+'hold/3')
            for kind in ('encoder','predictor','readout_old_data','readout_maze_data'):
                for repeat in (0,1):
                    budget.reserve_component(kind, case, repeat)
                with self.assertRaises(ValueError):
                    budget.reserve_component(kind, case, 0)
                with self.assertRaises(ValueError):
                    budget.reserve_component(kind, case, 2)
            budget.finish_source(case, None)
        self.assertEqual(len(budget.branches),504)
        self.assertEqual(sum(budget.source_ns.values()),105_600_000_000)
        self.assertEqual(len(budget.snapshots),24)
        self.assertEqual(len(budget.components),48)
        with self.assertRaises(ValueError):
            budget.start_source(0)
        with self.assertRaises(ValueError):
            budget.start_source(6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
