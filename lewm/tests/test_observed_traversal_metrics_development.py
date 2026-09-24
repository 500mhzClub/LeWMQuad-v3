import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.observed_traversal_scene_development import trials
from lewm.observed_traversal_metrics_development import reduce_traversal
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def fixture():
    count = 1200
    pose = np.tile([0., 0., .3, 0., 0., 0., 1.], (count, 1))
    pose[750:950, 0] = np.linspace(0., 1.2, 200)
    pose[950:, 0] = 1.2
    raw = {'timestamp_s': .002*np.arange(1, count+1), 'base_pose_world': pose,
           'base_twist_world': np.zeros((count, 6)), 'joint_position': np.zeros((count, 12)),
           'phase': np.array([0]*750+[1]*200+[2]*250), 'physics_contact': np.zeros(count, bool)}
    return trials()[0], raw, ArticulatedCollisionGeometry(URDF)


def evaluate(spec, raw, model, candidate_index=None, stop=None):
    decisions = [] if candidate_index is None else [{'pre_sample_index': candidate_index,
        'controller': {'status': 'ARRIVAL_CANDIDATE'}}]
    return reduce_traversal(spec, raw, 749, decisions,
        'FAILED_TIMEOUT' if candidate_index is None else 'ARRIVAL_CANDIDATE', stop, None, model)


def test_candidate_and_release_must_both_agree_with_evaluation_geometry():
    spec, raw, model = fixture()
    result = evaluate(spec, raw, model, 949)
    assert result['integration_success'] and result['physical_arrival']
    assert not result['false_arrival_candidate'] and result['trusted_graph_edges'] == 0
    # A premature observation cannot be repaired retroactively by later motion.
    result = evaluate(spec, raw, model, 800)
    assert result['physical_arrival'] and result['false_arrival_candidate']
    assert not result['integration_success']


def test_arrival_without_detection_contact_and_unsettled_release_are_separate():
    spec, raw, model = fixture()
    result = evaluate(spec, raw, model)
    assert result['physical_arrival_without_candidate'] and not result['integration_success']
    raw['physics_contact'][-1] = True
    result = evaluate(spec, raw, model, 949, 'DISALLOWED_CONTACT')
    assert result['candidate_geometry_agrees'] and not result['false_arrival_candidate']
    assert result['candidate_without_viable_release'] and not result['integration_success']
    raw['physics_contact'][:] = False
    raw['base_twist_world'][-1, 0] = .11
    result = evaluate(spec, raw, model, 949)
    assert not result['physical_checks']['release_motion']


def test_crossed_center_is_insufficient_for_whole_articulated_body():
    spec, raw, model = fixture()
    raw['base_pose_world'][950:, 0] = .85
    result = evaluate(spec, raw, model, 950)
    assert result['physical_checks']['sustained_center_crossing']
    assert not result['physical_checks']['whole_body_past_opening']
    assert result['false_arrival_candidate'] and not result['integration_success']
