"""Read-only five-configuration sensor-query diagnostic; no physical commands."""
import json
import time

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.continuous_startup_handoff_development import ContinuousStartupHandoff
from lewm.observed_setup_configuration_development import query_configuration
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_startup_observation_turn_development_v1 import json_same
from scripts.audit_go2_bounded_floor_robot_interface_native_box_reader_development_v1 import same
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_continuous_startup_handoff_development import IDENTITIES, SOURCES as HANDOFF_SOURCES
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_startup_observation_turn_development_v1 import OUTPUT, PROTOCOL, verify_extensions
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest
from scripts.startup_observation_turn_session_development import make_priors

SOURCES = HANDOFF_SOURCES + ('lewm/observed_setup_configuration_development.py',
    'lewm/tests/test_observed_setup_configuration_development.py',
    'scripts/probe_go2_observed_setup_configuration_development.py',
    'docs/go2_observed_setup_configuration_recorded_diagnostic_2026-09-06.md')


def main():
    bindings = {str((OUTPUT/p).relative_to(ROOT)): h for p,h in IDENTITIES.items()}
    verify_bindings(bindings)
    launch = json.loads((OUTPUT/'launch.json').read_text()); result = json.loads((OUTPUT/'result.json').read_text())
    bindings |= launch['source_sha256'] | launch['input_sha256'] | {
        str((OUTPUT/p).relative_to(ROOT)):h for p,h in result['artifact_sha256'].items()}
    sources = {p:digest(ROOT/p) for p in SOURCES}
    verify_bindings(bindings | sources); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    velocity, region = make_priors(1_500_000_000, launch['source_sha256'][PROTOCOL])
    admission = json.loads((OUTPUT/'startup_admission.json').read_text()); admission['identity'] = tuple(admission['identity'])
    owner = ContinuousStartupHandoff(ArticulatedCollisionGeometry(URDF), velocity_prior=velocity, region_prior=region, admission=admission)
    decisions = json.loads((OUTPUT/'startup_decisions.json').read_text())
    relatives = json.loads((OUTPUT/'relative_state_observations.json').read_text())
    for frame in range(result['rgbd_frames']):
        p,d = load_rgbd_observation(OUTPUT,frame); now = p['sensor_state']['decision_ns']
        row = owner.observe(p,d,load_fast_packet(OUTPUT,frame),now_ns=now)
        if row['terminal']: raise ValueError('recorded state owner failed: '+str(row))
        json_same(owner.relative_observation(now_ns=now),relatives[frame]['observer'])
        if frame < len(decisions): json_same(row['startup_decision'],decisions[frame]['decision'])
    owner.navigation_snapshot(now_ns=now)
    summaries = []
    for distance in (0., .25, .50, .75, 1.):
        args = (owner, [distance,0.,0.], np.eye(3), owner._memory._joints, 0.)
        start = time.perf_counter()
        actual = query_configuration(*args, now_ns=now, through_ns=3_300_000_000, reference='current_body', backend='compiled')
        elapsed = 1000*(time.perf_counter()-start)
        reference = query_configuration(*args, now_ns=now, through_ns=3_300_000_000, reference='current_body', backend='reference')
        same(actual, reference)
        rows = actual['primitives']
        summary = dict(forward_configuration_offset_m=distance,
            conditional_nonfloor_clear=[r['shape_id'] for r in rows if r['conditional_nonfloor_clearance']],
            setup_used=[r['shape_id'] for r in rows if r['supplied_clearance_used']],
            residuals_needing_observation=sum(len(r['residual_clearance_sources']) for r in rows),
            residuals_with_clearance=sum(bool(v) for r in rows for v in r['residual_clearance_sources']),
            obstacle_veto=[r['shape_id'] for r in rows if r['obstacle_veto_sources']],
            floor_penetration=[r['shape_id'] for r in rows if r['floor_penetration_sources']],
            observed_foot_contact_candidates=[r['shape_id'] for r in rows if r['observed_foot_contact_candidate']],
            compiled_reference_all_fields_exact=True, compiled_query_ms=elapsed,
            observation_bindings=actual['observation_bindings'], navigation_action_permitted=False)
        summaries.append(summary); print(json.dumps(summary,allow_nan=False),flush=True)
    verify_bindings(bindings | sources); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    print(json.dumps(dict(status='RECORDED_CONFIGURATION_QUERIES_COMPLETE', source_sha256=sources,
        configurations=summaries, initial_epoch_ns=velocity.anchor_ns, measured_ns=now,
        setup_expiry_ns=region.valid_until_ns, pose_scale_m=owner._memory._rays.fusion['position_error_scale_m'],
        source_scope='nine explicit new/predecessor development witnesses plus frozen startup inventory',
        scope='supplied rigid configurations, not dynamics predictions, swept motions or physical execution',
        original_result_unchanged=True, navigation_qualified=False),allow_nan=False),flush=True)


if __name__ == '__main__': main()
