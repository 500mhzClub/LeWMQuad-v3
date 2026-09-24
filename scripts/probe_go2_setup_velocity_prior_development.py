"""Conditional read-only prior initialization, budget and evaluator diagnosis."""
import json

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.setup_velocity_prior_development import SetupVelocityPrior, SetupVelocityIntegrator, SetupVelocityPlaneMemory
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings, URDF
from scripts.diagnose_go2_plane_memory_startup_development import cause_chain
from scripts.probe_go2_measured_plane_obstacle_memory_development import DIRECTORY, IDENTITIES, SOURCES
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest


PROTOCOL = 'docs/go2_setup_velocity_prior_recorded_diagnostic_2026-09-06.md'
EXTRA = ('lewm/setup_velocity_prior_development.py', 'lewm/setup_region_prior_development.py',
    'lewm/tests/test_setup_priors_development.py', 'scripts/probe_go2_setup_velocity_prior_development.py',
    'scripts/diagnose_go2_plane_memory_startup_development.py', PROTOCOL)


def main():
    bindings = {str((DIRECTORY / p).relative_to(ROOT)): h for p, h in IDENTITIES.items()}
    verify_bindings(bindings)
    launch = json.loads((DIRECTORY / 'launch.json').read_text())
    result = json.loads((DIRECTORY / 'result.json').read_text())
    reader = json.loads((DIRECTORY / 'native_box_reader_correction.json').read_text())
    assert reader['result']['interface_check_pass'] and result['rgbd_frames'] == 26
    bindings |= launch['source_sha256'] | launch['input_sha256'] | reader['reader_source_sha256']
    bindings |= {str((DIRECTORY / p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    development = {p: digest(ROOT / p) for p in (*SOURCES, *EXTRA)}
    verify_bindings(bindings); verify_bindings(development)
    print(json.dumps({'development_source_sha256': development}), flush=True)
    records = json.loads((DIRECTORY / 'relative_state_observations.json').read_text())
    assert len(records) == 26
    prior = SetupVelocityPrior((0, 0, 0), 1_500_000_000, (0., 0., 0.), .02, development[PROTOCOL])
    offline = SetupVelocityIntegrator(prior)
    memory = SetupVelocityPlaneMemory(ArticulatedCollisionGeometry(URDF), prior=prior,
        normal_error=.002, up_error=.001, plane_offset_error=.001, range_error_m=.001, beam_backend='compiled')
    stop = None; predictions = []
    try:
        for tick, record in enumerate(records):
            p, d = load_rgbd_observation(DIRECTORY, tick); now = p['sensor_state']['decision_ns']
            row = offline.observe(p, record['observer']); predictions.append(row)
            if stop is None:
                try: memory.observe(p, d, record['observer'], now_ns=now)
                except SensorContractError as error:
                    stop = dict(tick=tick, measured_ns=now, causes=cause_chain(error), fault_latched=memory.failed)
                else:
                    assert memory._rays.fusion == row
                    assert memory._rays.latest_frame['position_scale_m'] == row['position_error_scale_m']
            transport = row['initial_velocity_prior_transport']
            print(json.dumps(dict(tick=tick, depth_rank=row['depth_rank'],
                prior_position_radius_m=transport['position_radius_m'],
                inherited_position_scale_m=row['inherited_position_error_scale_m'],
                combined_position_scale_m=row['position_error_scale_m'],
                budget_usable=row['usable_under_declared_proxy_budget'],
                post_memory_stop_diagnostic_only=stop is not None)), flush=True)
        # Evaluator-only comparison occurs after all policy-side computations.
        with np.load(DIRECTORY / 'physics_trace.npz', allow_pickle=False) as z:
            ns = np.rint(z['timestamp_s'] * 1e9).astype(np.int64)
            indices = np.searchsorted(ns, [p['measured_ns'] for p in predictions])
            assert np.array_equal(ns[indices], [p['measured_ns'] for p in predictions])
            pose = z['base_pose_world'][indices]; twist = z['base_twist_world'][indices]
        R0 = rotation_xyzw(pose[0, 3:])
        initial_velocity = R0.T @ twist[0, :3]
        initial_error = float(np.linalg.norm(initial_velocity - prior.mean_initial_body_m_s))
        actual_position = (pose[:, :3] - pose[0, :3]) @ R0
        errors = np.linalg.norm(np.array([p['position_initial_body_m'] for p in predictions]) - actual_position, axis=1)
        scales = np.array([p['position_error_scale_m'] for p in predictions])
        print(json.dumps(dict(status='CONDITIONAL_SETUP_PRIOR_RECORDED_DIAGNOSTIC_COMPLETE', frames=len(predictions),
            memory_stop=stop, proposed_initial_velocity_ball_contains_evaluator=initial_error <= prior.radius_m_s,
            initial_velocity_reference_initial_body_m_s=initial_velocity.tolist(), initial_velocity_error_m_s=initial_error,
            maximum_position_error_m=float(errors.max()), final_position_error_m=float(errors[-1]),
            position_error_at_memory_stop_m=float(errors[stop['tick']]) if stop else None,
            combined_scale_at_memory_stop_m=float(scales[stop['tick']]) if stop else None,
            proxy_exceedances=int((errors > scales).sum()), final_combined_scale_m=float(scales[-1]),
            initial_setup_independently_validated=False, region_prior_injected=False,
            original_no_prior_failure_preserved=True, navigation_qualified=False)), flush=True)
    finally:
        verify_bindings(bindings); verify_bindings(development)
    print('ALL_BINDINGS_VERIFIED_BEFORE_AND_AFTER', flush=True)


if __name__ == '__main__': main()
