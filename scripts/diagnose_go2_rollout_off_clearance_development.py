"""Recorded current-clearance stalls, with evaluator-only native wall distances."""
import argparse
from collections import Counter
import json
import numpy as np
from lewm.physical_execution_development import rotation_xyzw
from scripts.compare_continuous_navigation_arms_development import path,read


def native_wall_clearance(position,geometries):
    distances=[]
    for g in geometries:
        if g['geom_type']!='BOX':continue
        quat=np.asarray(g['quaternion_world_wxyz'])
        R=rotation_xyzw(quat[[1,2,3,0]])
        if not np.allclose(R[2,:2],0.,atol=1e-8):
            raise ValueError('horizontal wall footprint diagnostic requires upright boxes')
        local=(np.asarray(position)-g['position_world_m'])@R
        outside=np.maximum(np.abs(local[:2])-np.asarray(g['data'][:2])/2,0.)
        distances.append(float(np.linalg.norm(outside)))
    return min(distances) if distances else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    parser.add_argument('--action-reserve',action='store_true')
    args=parser.parse_args();root=path(args.root_name)
    clear_key='current_action_reserve_clear' if args.action_reserve else 'current_nominal_disk_clear'
    plans=[p for p in read(root,'planning.json') if 'selection' in p]
    blocked=[p for p in plans if not p['selection'][clear_key]]
    requests=read(root,'requests.json')
    frames={r['frame']:r for r in read(root,'native/in_memory_camera_observations.json')['frames']}
    geometries=read(root,'native/camera_setup_identity.json')['environment']['physical_geometries']
    checks=[]
    if blocked:
        with np.load(root/'native/physics_trace.npz',allow_pickle=False) as arrays:
            physics=arrays['base_pose_world']
        for p in (blocked[0],blocked[-1]):
            checks.append(dict(frame=p['frame'],measured_ns=p['measured_ns'],
                stored_clearance_m=p['selection']['current_stored_clearance_m'],
                native_body_centre_wall_clearance_m=native_wall_clearance(
                    physics[frames[p['frame']]['physical_sample_index'],:3],geometries)))
    recovery={r['view_recovery']['trigger_ns'] for r in requests if r.get('view_recovery')
        and r['view_recovery']['source']=='actual_translation_veto'}
    result=dict(blocked_plans=len(blocked),selected_plans=len(plans),
        first_blocked_frame=blocked[0]['frame'] if blocked else None,
        consecutive_through_terminal=bool(blocked) and all(not p['selection'][clear_key]
            for p in plans if p['frame']>=blocked[0]['frame']),
        blocked_actions=dict(Counter(p['action'] for p in blocked)),clearance_checks=checks,
        last_translation_request_ns=max((r['now_ns'] for r in requests if any(r['requested_command'][:2])),default=None),
        actual_translation_veto_recoveries=len(recovery),native_geometry_evaluation_only=True,
        map_clearance_discrepancy_cause_isolated=False,turn_only_escape_safety_established=False,
        checked_clearance_field=clear_key,required_clearance_m=.48 if args.action_reserve else .45)
    filename='current_action_reserve_diagnostic_v1.json' if args.action_reserve else 'rollout_off_current_clearance_diagnostic_v1.json'
    with (root/filename).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
