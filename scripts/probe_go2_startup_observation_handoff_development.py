"""Read-only terminal-controller memory diagnostic; no restart or physics."""
import json

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.startup_observation_turn_development import StartupObservationTurn
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_startup_observation_turn_development_v1 import json_same
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_startup_observation_turn_development_v1 import OUTPUT, PROTOCOL, verify_extensions
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest
from scripts.startup_observation_turn_session_development import make_priors


def main():
    launch = json.loads((OUTPUT/'launch.json').read_text()); result = json.loads((OUTPUT/'result.json').read_text())
    audit = json.loads((OUTPUT/'raw_artifact_audit.json').read_text())
    if audit['status'] != 'RAW_REPLAY_COMPLETE' or not audit['local_observation_turn_success']:
        raise ValueError('verified completed local observation turn required for this handoff diagnostic')
    bindings = launch['source_sha256'] | launch['input_sha256'] | {
        str((OUTPUT/p).relative_to(ROOT)): h for p, h in result['artifact_sha256'].items()}
    bindings |= {str((OUTPUT/p).relative_to(ROOT)): digest(OUTPUT/p) for p in ('launch.json', 'result.json', 'raw_artifact_audit.json')}
    name = 'scripts/probe_go2_startup_observation_handoff_development.py'; bindings[name] = digest(ROOT/name)
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    velocity, region = make_priors(1_500_000_000, launch['source_sha256'][PROTOCOL])
    admission = json.loads((OUTPUT/'startup_admission.json').read_text()); admission['identity'] = tuple(admission['identity'])
    model = StartupObservationTurn(ArticulatedCollisionGeometry(URDF), velocity_prior=velocity, region_prior=region, admission=admission)
    decisions = json.loads((OUTPUT/'startup_decisions.json').read_text())
    relatives = json.loads((OUTPUT/'relative_state_observations.json').read_text())
    for item in decisions:
        frame = item['observation_index']; p, d = load_rgbd_observation(OUTPUT, frame)
        now = p['sensor_state']['decision_ns']
        json_same(model.observe(p, d, relatives[frame]['observer'], now_ns=now), item['decision'])
    # Query the exact last decision state. Never feed postterminal observations,
    # restart the controller, extend its region or change its recorded outcome.
    query = model.memory.query_current_primitives(now_ns=now)
    ids = np.asarray(query['shape_ids'])
    fields = ('conditional_clearance', 'foot_contact_candidate', 'non_floor_conflict', 'floor_penetration')
    report = dict(status='TERMINAL_MEMORY_HANDOFF_DIAGNOSTIC_COMPLETE', controller_status=model.status,
        decision_ns=now, combined_position_scale_m=model.memory._rays.fusion['position_error_scale_m'],
        view_count=len(query['views']), views=query['views'], **{key: ids[query[key]].tolist() for key in fields},
        current_measured_posture_only=query['current_measured_posture_only'],
        region_expiry_ns=region.valid_until_ns, contact_permitted=query['contact_permitted'],
        navigation_qualified=False, full_mission_success=False, diagnostic_source_sha256=bindings[name],
        scope='read-only exact controller-terminal memory; no postterminal replay, continuation or outcome change')
    verify_bindings(bindings); verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__': main()
