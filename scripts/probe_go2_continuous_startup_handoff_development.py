"""New read-only continuous-state replay of the saved startup/tail observations.

This is not a rerun, resume, old-controller restart or rescore of its outcome.
The new owner reproduces the old startup decisions, then updates the SAME memory
from the three saved tail frames; it neither executes nor approves navigation.
"""
import json

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.continuous_startup_handoff_development import ContinuousStartupHandoff
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.setup_clearance_partition_development import current_body_setup_partition
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_startup_observation_turn_development_v1 import json_same
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_startup_observation_turn_development_v1 import OUTPUT, PROTOCOL, verify_extensions
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest
from scripts.startup_observation_turn_session_development import make_priors

IDENTITIES = {
    'launch.json': '81d4caf731b0b607c55f967601e309b252a651cb95fb76b1559e4c17cbf12f0d',
    'result.json': 'cc299c1a1aea1bad7162da8283929ea41daa3669fc8a35e47124ac85cf21feb5',
    'raw_artifact_audit.json': '2f4fa9785cf1aaf7cad848a080fa1754f581df25508b97ad40089f721f4f20fc',
}
SOURCES = ('lewm/continuous_startup_handoff_development.py',
           'lewm/tests/test_continuous_startup_handoff_development.py',
           'lewm/setup_clearance_partition_development.py',
           'lewm/tests/test_setup_clearance_partition_development.py',
           'scripts/probe_go2_continuous_startup_handoff_development.py')


def main():
    bindings = {str((OUTPUT/p).relative_to(ROOT)): h for p, h in IDENTITIES.items()}
    verify_bindings(bindings)
    launch = json.loads((OUTPUT/'launch.json').read_text()); result = json.loads((OUTPUT/'result.json').read_text())
    bindings |= launch['source_sha256'] | launch['input_sha256'] | {
        str((OUTPUT/p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    sources = {p: digest(ROOT/p) for p in SOURCES}
    verify_bindings(bindings | sources); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    velocity, region = make_priors(1_500_000_000, launch['source_sha256'][PROTOCOL])
    admission = json.loads((OUTPUT/'startup_admission.json').read_text()); admission['identity'] = tuple(admission['identity'])
    owner = ContinuousStartupHandoff(ArticulatedCollisionGeometry(URDF), velocity_prior=velocity,
        region_prior=region, admission=admission)
    memory, observer = owner._memory, owner._relative
    decisions = json.loads((OUTPUT/'startup_decisions.json').read_text())
    relatives = json.loads((OUTPUT/'relative_state_observations.json').read_text())
    rows = []
    for frame in range(result['rgbd_frames']):
        p, d = load_rgbd_observation(OUTPUT, frame); fast = load_fast_packet(OUTPUT, frame)
        now = p['sensor_state']['decision_ns']; row = owner.observe(p, d, fast, now_ns=now)
        if row['terminal']: raise ValueError('new handoff failed on saved frames: '+str(row))
        json_same(owner.relative_observation(now_ns=now), relatives[frame]['observer'])
        if frame < len(decisions): json_same(row['startup_decision'], decisions[frame]['decision'])
        elif row['startup_decision'] is not None: raise ValueError('terminal startup controller called again')
        if owner._memory is not memory or owner._relative is not observer: raise ValueError('observer/memory replaced')
        rows.append({k:v for k,v in row.items() if k != 'startup_decision'})
    state = owner.navigation_snapshot(now_ns=now)
    partition = current_body_setup_partition(owner, now_ns=now)
    verify_bindings(bindings | sources); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    print(json.dumps(dict(status='CONTINUOUS_HANDOFF_RECORDED_REPLAY_COMPLETE', rows=rows,
        startup_decisions_exact=len(decisions), relative_observations_exact=len(relatives),
        same_observer_and_memory=True, initial_epoch_ns=state['initial_epoch_ns'], final_ns=state['measured_ns'],
        consumed_observations=state['consumed_observations'], gyro_intervals=state['gyro_intervals'],
        retained_view_ns=state['retained_view_ns'], latest_view_ns=state['latest_view_ns'],
        fusion=state['fusion'], observed_current_posture=state['observed_current_posture'],
        current_body_partition={k:v for k,v in partition.items() if k != 'per_shape'},
        supplied_region_active=state['supplied_region_active'], source_sha256=sources,
        old_result_unchanged=True, navigation_action_permitted=False, navigation_qualified=False,
        scope='new read-only state-owner replay, including saved tail; not physical continuation or mission success'), allow_nan=False), flush=True)


if __name__ == '__main__': main()
