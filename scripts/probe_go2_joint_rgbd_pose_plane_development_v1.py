"""Two fixed finite-difference steps on saved development sensors, no motion."""
import json
import time

import cv2
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.paired_rgbd_physical_plane_development import PairedRGBDPhysicalPlane
from lewm.raw_complementary_rgbd_sensitivity_development import RawComplementaryRGBDSensitivity, SHAPES
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.rgbd_shadow_motion_development import POINT_HYPOTHESES
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_fresh_fused_maze_development_v1 import exact
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.fresh_maze_session_development import priors
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.probe_go2_rgbd_physical_configuration_evidence_development_v1 import OUTPUT as PREVIOUS
from scripts.run_go2_fresh_fused_maze_development_v1 import OUTPUT as INPUT, PROTOCOL as INPUT_PROTOCOL
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = ROOT/'.generated/go2_joint_rgbd_pose_plane_development_v1_attempt_001'
PROTOCOL = 'docs/go2_joint_rgbd_pose_plane_development_v1_2026-09-06.md'
SEEDS = (PROTOCOL, 'scripts/probe_go2_joint_rgbd_pose_plane_development_v1.py',
         'lewm/tests/test_raw_complementary_rgbd_sensitivity_development.py')
IDENTITIES = {'launch.json': '91df22609d12ed954ca66efa122bb92a48586c2c5d634fbee478d12c752c7c58',
    'result.json': 'd75e8a9f044fa22b5017bb17a5a796143cab30ace052230193cf9735718a85c3',
    'reference_audit.json': 'f915202645ffc6817d6222b259a22e3c77ae06655cd516fe8c0d606dd83d1059'}
SOURCES = ('persistent_depth_scale_0p1percent', 'persistent_depth_offset_1mm',
           'persistent_yaw_gyro_bias_1mrad_per_second', 'post_anchor_force_y_bias_1cm_per_second2',
           'initial_velocity_y_prior_mean_1cm_per_second')
STEPS = (.01, .005)
PLANE_ERRORS = dict(normal_error=.002, up_error=.001, plane_offset_error=.001)


def plain(value):
    """Unknown numerical relations remain explicit nulls, never zero evidence."""
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def loadings(policy, depth, fast):
    result = {k: np.zeros((*shape, len(SOURCES))) for k, shape in SHAPES.items()}
    result['depth_m'][..., 0] = .001*depth['depth_m']
    result['depth_m'][..., 1] = .001*depth['valid']
    result['gyro'][:, 2, 2] = .001; result['fast_gyro'][:, 2, 2] = .001
    after = policy['sensor_state']['sensed']['specific_force']['measured_ns'] > 1_500_000_000
    result['specific_force'][after, 1, 3] = .01
    result['initial_velocity'][1, 4] = .01
    return result


def preflight():
    bindings = {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(bindings)
    old = read_json(PREVIOUS, 'launch.json'); verify(old)
    result = read_json(PREVIOUS, 'result.json'); audit = read_json(PREVIOUS, 'reference_audit.json')
    verify_bindings(audit['auditor_source_sha256'])
    bindings |= {str((PREVIOUS/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    definition = old | dict(source_sha256=discover_sources(SEEDS, old['source_sha256']),
        input_sha256=old['input_sha256'] | bindings | audit['auditor_source_sha256'],
        error_sources=list(SOURCES), finite_difference_steps=list(STEPS),
        plane_error_hypotheses=PLANE_ERRORS, diagnostic_protocol=PROTOCOL,
        scope='fixed shared raw-error finite differences; no calibration, physical action or covariance-based permission')
    verify(definition); return definition


def run(step, launch):
    velocity, _ = priors(launch['source_sha256'][INPUT_PROTOCOL])
    model = RawComplementaryRGBDSensitivity(SOURCES, prior=velocity, hypotheses=POINT_HYPOTHESES, difference_step=step)
    planes = PairedRGBDPhysicalPlane(model); saved = read_json(INPUT/'mission', 'task_decisions.json')
    rows = []; start = time.perf_counter_ns()
    for i, expected in enumerate(saved):
        p, d = load_rgbd_observation(INPUT/'mission', i); f = load_fast_packet(INPUT/'mission', i)
        now = p['sensor_state']['decision_ns']
        result = model.observe(p, d, f, loadings(p, d, f))
        exact(result['nominal_fusion'], expected['controller']['sensor_fusion'])
        exact(result['nominal_raw_depth_state']['motion'], expected['controller']['raw_depth_motion'])
        if i == 0: planes.retain('initial')
        # Keep useful evidence but avoid duplicating every nominal raw surface.
        rows.append(plain({k: v for k, v in result.items() if k not in ('nominal_raw_depth_state', 'nominal_point_state')}))
        if i % 20 == 0: print(json.dumps(dict(step=step, replayed=i+1, categorical_change_seen=bool(model.branch_changes))), flush=True)
    planes.retain('terminal')
    geometry = ArticulatedCollisionGeometry(URDF); q = p['sensor_state']['sensed']['joints']['values'][-1, :12]
    gaps = {label: plain(planes.query(label, geometry, q, [0., 0., 0.], np.eye(3), now_ns=now, **PLANE_ERRORS))
            for label in ('initial', 'terminal')}
    return dict(step=step, exact_nominal_frames=len(rows), member_models=len(model.models), observations=rows,
        physical_gap_relations=gaps, categorical_changes=model.branch_changes, elapsed_wall_ms=(time.perf_counter_ns()-start)/1e6,
        runtime_physics_executed=False, uncertainty_model_calibrated=False, navigation_action_permitted=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('fresh fixed diagnostic only')
    cv2.setNumThreads(1); launch = preflight()
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json', launch)
    completed = []
    try:
        for i, step in enumerate(STEPS):
            name = f'step_{i:02d}.json'; result = run(step, launch)
            write_json(OUTPUT/name, result); completed.append(name)
        verify(launch)
        result = dict(status='TWO_STEP_JOINT_RGBD_POSE_PLANE_DIAGNOSTIC_COMPLETE',
            artifact_sha256={name: digest(OUTPUT/name) for name in completed},
            independent_experimental_trials=0, physics_executed=False, calibrated_error_bounds=False,
            navigation_qualified=False, goal_achieved=False)
        write_json(OUTPUT/'result.json', result); print(result['status'], flush=True)
    except Exception as error:
        reasons = []
        while error is not None:
            reasons.append(str(error)); error = error.__cause__
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_JOINT_RGBD_DIAGNOSTIC_FAILURE', reasons=reasons,
            completed_artifacts={n: digest(OUTPUT/n) for n in completed}))
        raise


if __name__ == '__main__': main()
