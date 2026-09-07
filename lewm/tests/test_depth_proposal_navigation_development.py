import ast
from copy import deepcopy
import hashlib
import inspect
import textwrap

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import from_native_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.depth_floor_hold_navigation_development import ObservedDepthFloor
from lewm.depth_proposal_navigation_development import DepthProposalNavigation, DepthProposalTraversal
from lewm.depth_supported_exit_candidates_development import observe_depth_exit_candidates
from lewm.rgbd_fused_navigation_development import RGBDFusedTraversal
from lewm.rgbd_inertial_ray_memory_development import RGBDInertialRayMemory
from lewm.observable_approach_development import ObservableApproachRegions
from lewm.rgbd_fused_navigation_development import PreparedRayView
from lewm.whole_task_navigation_development import WholeTaskNavigation
from lewm.tests.test_rgbd_fused_navigation_development import frames
from lewm.tests.test_rgbd_inertial_fusion_development import prior, hypotheses
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def gray_frames(count):
    for p, f, d, now in frames(count):
        p['image']['rgb'] = np.repeat(p['image']['rgb'][:, :, :1], 3, axis=2)
        d['rgb_sha256'] = hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
        yield p, f, d, now


def setup_proposals():
    p, f, d, now = next(gray_frames(1))
    geometry = ArticulatedCollisionGeometry(URDF)
    owner = RGBDInertialRayMemory(prior=prior(), hypotheses=hypotheses())
    row = owner.observe(p, d, f, now_ns=now)
    regions = ObservableApproachRegions(geometry)
    regions.memory = PreparedRayView(owner)
    regions.memory.stage(p, d, row)
    regions.observe(p, d, row['depth_state'], now_ns=now)
    ground = ObservedDepthFloor(regions).begin(p, now_ns=now)
    return p, d, ground, now


def propose(p, d, ground, now):
    return observe_depth_exit_candidates(p, d, ground, now_ns=now,
                                        observation_id=hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest())


def test_palette_independence_uses_measured_depth_not_synthetic_rgb_relabel():
    p, d, ground, now = setup_proposals()
    before = deepcopy(p)
    expected = propose(p, d, ground, now)
    assert expected['candidate_rows'] and not expected['color_segmentation_used']
    np.testing.assert_array_equal(p['image']['rgb'], before['image']['rgb'])
    p['image']['rgb'][:] = [180, 20, 210]
    d['rgb_sha256'] = hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    changed = propose(p, d, ground, now)
    np.testing.assert_array_equal(expected['angle_radial_support_counts'], changed['angle_radial_support_counts'])
    assert expected['depth_sha256'] == changed['depth_sha256']
    assert all(not c['qualified_exit'] and not c['qualified_traversal'] for c in changed['candidate_rows'])


@pytest.mark.parametrize('kind', ['unknown', 'vertical_wall', 'raised_plane'])
def test_unknown_walls_and_different_height_planes_never_become_floor(kind):
    p, d, ground, now = setup_proposals()
    if kind == 'unknown':
        d['depth_m'][:] = 0
        d['valid'][:] = False
    elif kind == 'vertical_wall':
        d = from_native_depth(np.full((480, 640), 2., np.float32), p,
                              measured_ns=now, available_ns=now, now_ns=now)
    else:
        # Move the original floor's depth returns towards the camera; they no
        # longer lie on the observed ground plane supplied to this test.
        d['depth_m'][d['valid']] *= .6
    row = propose(p, d, ground, now)
    assert not row['candidate_rows'] and not row['absence_means_closed']


def test_invalid_rays_remove_support_and_are_not_bridged_by_plane_projection():
    p, d, ground, now = setup_proposals()
    original = propose(p, d, ground, now)
    d['valid'][:, 160:480] = False
    d['depth_m'][:, 160:480] = 0
    masked = propose(p, d, ground, now)
    assert masked['observed_point_count'] < original['observed_point_count']
    assert np.all(masked['angle_radial_support_counts'] <= original['angle_radial_support_counts'])
    assert not any(abs(c['bearing_body_rad']) < .1 for c in masked['candidate_rows'])


@pytest.mark.parametrize('fault', ['clock', 'ground_source', 'rgb_binding', 'normal'])
def test_proposal_sensor_ground_contracts(fault):
    p, d, ground, now = setup_proposals()
    if fault == 'clock': ground['decision_ns'] -= 1
    elif fault == 'ground_source': ground['estimator_mode'] = 'supplied_world_plane'
    elif fault == 'rgb_binding': d['rgb_sha256'] = '0'*64
    else: ground['up_current_body'] = [0., 0., 2.]
    with pytest.raises(SensorContractError): propose(p, d, ground, now)


def test_grayscale_full_controller_reaches_traversal_with_raw_weak_depth():
    c = DepthProposalNavigation(ArticulatedCollisionGeometry(URDF), memory_arm='episodic',
                                prior=prior(), hypotheses=hypotheses())
    proposal_ticks = []
    for tick, (p, f, d, now) in enumerate(gray_frames(24)):
        row = c.observe_rgbd(p, f, d, now_ns=now)
        if row['exit_proposal_evidence']:
            proposal_ticks.append(tick)
        assert not row['terminal']
        assert c._context is None and c._proposal_depth is None
    assert proposal_ticks == [3, 22]
    assert isinstance(c.child, DepthProposalTraversal) and row['child']['status'] == 'TRAVERSING'
    assert row['raw_depth_motion']['translation_previous_body_m'] is None
    assert row['requested_command'][0] > 0 and row['trusted_graph_edges'] == 0
    assert c.sensor_memory.state.depth.orientation.samples_integrated == 23*50


@pytest.mark.parametrize('original,successor,target', [
    (WholeTaskNavigation._observe, DepthProposalNavigation._observe, 'self._propose'),
    (RGBDFusedTraversal._fused_step, DepthProposalTraversal._fused_step, 'self.navigator._propose')])
def test_copied_decisions_change_only_proposal_provider(original, successor, target):
    before = ast.parse(textwrap.dedent(inspect.getsource(original)))
    after = ast.parse(textwrap.dedent(inspect.getsource(successor)))
    class Replace(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.parse(target, mode='eval').body if node.id == 'observe_exit_candidates' else node
    expected = Replace().visit(before)
    assert ast.dump(expected, include_attributes=False) == ast.dump(after, include_attributes=False)
