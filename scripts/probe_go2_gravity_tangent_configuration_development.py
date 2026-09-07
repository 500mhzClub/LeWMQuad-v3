"""Separate exploratory sensor-gravity-tangent configuration replay; no physics."""
import json

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.continuous_startup_handoff_development import ContinuousStartupHandoff
from lewm.observed_setup_configuration_development import query_configuration
from lewm.observed_turn_region_development import gravity_basis
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_startup_observation_turn_development_v1 import json_same
from scripts.audit_go2_bounded_floor_robot_interface_native_box_reader_development_v1 import same
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_continuous_startup_handoff_development import IDENTITIES
from scripts.probe_go2_observed_setup_configuration_development import SOURCES as PREVIOUS_SOURCES
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_startup_observation_turn_development_v1 import OUTPUT, PROTOCOL, verify_extensions
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest
from scripts.startup_observation_turn_session_development import make_priors

SOURCES = PREVIOUS_SOURCES + ('scripts/probe_go2_gravity_tangent_configuration_development.py',
    'docs/go2_gravity_tangent_configuration_recorded_diagnostic_2026-09-06.md')


def main():
    bindings = {str((OUTPUT/p).relative_to(ROOT)):h for p,h in IDENTITIES.items()}
    verify_bindings(bindings)
    launch = json.loads((OUTPUT/'launch.json').read_text()); result = json.loads((OUTPUT/'result.json').read_text())
    bindings |= launch['source_sha256'] | launch['input_sha256'] | {
        str((OUTPUT/p).relative_to(ROOT)):h for p,h in result['artifact_sha256'].items()}
    sources = {p:digest(ROOT/p) for p in SOURCES}
    verify_bindings(bindings | sources); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    velocity, region = make_priors(1_500_000_000,launch['source_sha256'][PROTOCOL])
    admission = json.loads((OUTPUT/'startup_admission.json').read_text()); admission['identity'] = tuple(admission['identity'])
    owner = ContinuousStartupHandoff(ArticulatedCollisionGeometry(URDF),velocity_prior=velocity,region_prior=region,admission=admission)
    decisions = json.loads((OUTPUT/'startup_decisions.json').read_text())
    relatives = json.loads((OUTPUT/'relative_state_observations.json').read_text())
    for frame in range(result['rgbd_frames']):
        p,d = load_rgbd_observation(OUTPUT,frame); now = p['sensor_state']['decision_ns']
        row = owner.observe(p,d,load_fast_packet(OUTPUT,frame),now_ns=now)
        if row['terminal']: raise ValueError('saved handoff replay failed')
        json_same(owner.relative_observation(now_ns=now),relatives[frame]['observer'])
        if frame<len(decisions): json_same(row['startup_decision'],decisions[frame]['decision'])
    owner.navigation_snapshot(now_ns=now)
    up = owner._memory._rays.latest_frame['evidence']['up']; tangent = gravity_basis(up)[:,0]
    rows = []
    for distance in (0.,.25,.50,.75,1.):
        args = (owner,distance*tangent,np.eye(3),owner._memory._joints,0.)
        actual = query_configuration(*args,now_ns=now,through_ns=3_300_000_000,reference='current_body',backend='compiled')
        reference = query_configuration(*args,now_ns=now,through_ns=3_300_000_000,reference='current_body',backend='reference')
        same(actual,reference)
        shapes = actual['primitives']
        row = dict(tangent_offset_m=distance,conditional_nonfloor_clear=[s['shape_id'] for s in shapes if s['conditional_nonfloor_clearance']],
            setup_used=[s['shape_id'] for s in shapes if s['supplied_clearance_used']],
            residuals_needing_observation=sum(len(s['residual_clearance_sources']) for s in shapes),
            residuals_with_clearance=sum(bool(v) for s in shapes for v in s['residual_clearance_sources']),
            obstacle_veto=[s['shape_id'] for s in shapes if s['obstacle_veto_sources']],
            floor_penetration=[s['shape_id'] for s in shapes if s['floor_penetration_sources']],
            observed_foot_contact_candidates=[s['shape_id'] for s in shapes if s['observed_foot_contact_candidate']],
            compiled_reference_all_fields_exact=True,observation_bindings=actual['observation_bindings'],
            ground_support_permission=False,navigation_action_permitted=False)
        rows.append(row); print(json.dumps(row,allow_nan=False),flush=True)
    verify_bindings(bindings | sources); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    print(json.dumps(dict(status='GRAVITY_TANGENT_CONFIGURATION_DIAGNOSTIC_COMPLETE',source_sha256=sources,
        up_current_body=up.tolist(),forward_tangent_body=tangent.tolist(),body_forward_up_component=float(up[0]),
        tangent_forward_up_component=float(tangent@up),configurations=rows,
        scope='exploratory new rigid-configuration diagnostic; original body-axis negatives preserved; no physics or rescore',
        navigation_qualified=False),allow_nan=False),flush=True)


if __name__ == '__main__': main()
