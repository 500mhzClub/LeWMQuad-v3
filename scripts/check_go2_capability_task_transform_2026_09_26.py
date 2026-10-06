"""Zero-physics structural check of exactly the registered 180 episodes.

The user's 26 September instruction explicitly permits this structural-only
check of the new sealed set. No rendering, navigation, statistics by layout,
or disclosure of sealed geometry; legacy sealed material is never accessed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from lewm import decision_headroom_json_v42_development as output
from lewm.decision_headroom_reference_development import ReferenceGeometry
from lewm.navigation_capability_target_reference_development import nominal_targets, world_target, settled_task_cues, cue_world_xy
from lewm.physical_execution_development import rotation_xyzw
from scripts.generate_go2_navigation_capability_sets_development import walls_for_reference, occluded

REPO=Path(__file__).resolve().parents[1]


def main():
    protocol=json.loads((REPO/'docs/go2_navigation_capability_preregistration_v1_2026-09-25.json').read_text())
    base=Path(protocol['output_root']);output.install(base)
    registry=json.loads((base/'registry.json').read_text())
    root=base/'target_reference_correctness_c1_2026-09-26';root.mkdir(exist_ok=False)
    # One observed initialization displacement, not an outcome-selected state.
    pilot=base/'runs/v0_pilot_C0_dev00_ep0_attempt002'
    original=json.loads((pilot/'episode.json').read_text())
    with np.load(pilot/'native/physics_trace.npz',allow_pickle=False) as a:
        origin=a['base_pose_world'][749].copy()
    yaw=original['home_se2_world'][2]
    c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    offset=R.T @ (origin[:3]-np.array([*original['home_se2_world'][:2],origin[2]]))
    tilt=R.T @ rotation_xyzw(origin[3:])
    counts={role:dict(episodes=0,structural_pass=0,nominal_reference_pass=0,
                     pre_fix_settling_contract_failures=0,post_fix_target_checks=0,post_fix_failures=0) for role in ('dev_tune','validation','sealed_test')}
    maxima=dict(home_translation_m=0.,goal_instruction_difference_m=0.,post_fix_world_error_m=0.)
    def read(binding,role):
        path=Path(binding['path'])
        expected=base/'sets'/('sealed_test_v1' if role=='sealed_test' else role)
        assert path.parent==expected and path.suffix=='.json'
        data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==binding['sha256']
        return json.loads(data)
    for entry in registry['entries']:
        role=entry['role'];spec=read(entry['maze'],role)
        assert spec['data_role']==role
        walls=walls_for_reference(spec)
        for binding in entry['episodes']:
            e=read(binding,role);assert e['role']==role
            assert e['maze_id']==spec['layout_index']
            counts[role]['episodes']+=1
            start=np.asarray(e['home_se2_world'][:2]);target=np.asarray(e['beacon_xy_world'])
            cfg=protocol['definitions']['spl']
            geo=ReferenceGeometry(walls,protocol['generator']['world_bounds_xy_m'],target,
                radius_m=cfg['inflation_radius_m'],clearance_m=cfg['additional_clearance_m'],resolution_m=cfg['grid_resolution_m'])
            distance=geo.distance_and_heading(start)
            valid=(float(min(geo.footprint_clearance(start),geo.footprint_clearance(target)))+geo.radius>=.5
                   and occluded(start,target,walls) and distance['valid']
                   and distance['distance_m']>protocol['episodes']['minimum_start_beacon_geodesic_m']
                   and abs(distance['distance_m']-e['shortest_outbound_m'])<1e-9
                   and abs(distance['distance_m']-e['shortest_return_m'])<1e-9)
            assert valid, 'Registered episode structural rule failure (sealed details withheld)'
            counts[role]['structural_pass']+=1
            targets=nominal_targets(e)
            assert all(np.allclose(targets[p],world_target(e,p),atol=1e-9,rtol=0) for p in targets)
            counts[role]['nominal_reference_pass']+=1
            c,s=np.cos(e['home_se2_world'][2]),np.sin(e['home_se2_world'][2])
            rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            settled=np.array([*start,origin[2]])+rotation@offset
            actual_rotation=rotation@tilt
            desired=actual_rotation.T @ (np.array([*target,origin[2]])-settled)
            error=float(np.linalg.norm(desired[:2]-e['mission']['goal_initial_body_xy_m']))
            home_error=float(np.linalg.norm(settled[:2]-start))
            maxima['home_translation_m']=max(maxima['home_translation_m'],home_error)
            maxima['goal_instruction_difference_m']=max(maxima['goal_instruction_difference_m'],error)
            counts[role]['pre_fix_settling_contract_failures']+=int(max(home_error,error)>1e-9)
            for position, orientation in [(np.array([*start,origin[2]]),rotation),(settled,actual_rotation)]:
                cues=settled_task_cues(e,position,orientation)
                for phase,key in [('OUTBOUND','goal_initial_body_xy_m'),('RETURN','return_initial_body_xy_m')]:
                    reconstructed=cue_world_xy(cues[key],position,orientation)
                    residual=float(np.linalg.norm(reconstructed-world_target(e,phase)))
                    counts[role]['post_fix_target_checks']+=1
                    counts[role]['post_fix_failures']+=int(residual>1e-9)
                    maxima['post_fix_world_error_m']=max(maxima['post_fix_world_error_m'],residual)
                    assert residual<=1e-9, 'Post-fix target-reference contract failed (sealed details withheld)'

    assert sum(v['episodes'] for v in counts.values())==180
    result=dict(schema='navigation_capability_reference_structural_check.v1',physics_steps=0,
        counts=counts,all_registered_episode_geometry_valid=True,replacements_needed=0,
        tolerance_m=1e-9,pre_fix_reference_contract_passed=False,
        pre_fix_settling_test='Same source initialization displacement expressed under each registered start transform; a contract test, not measured dynamics of unrun episodes.',
        maxima=maxima,confirmed_affected_runs='All five pilots on 00/0 share this origin displacement',
        potentially_affected_episodes='All 180; actual settling displacement outside pilot 00/0 is not measured',
        sealed_access='Explicit user-authorized structural check only; aggregate counts only retained',
        post_fix_test_pending=False,post_fix_passed=True,controller_and_tracker_code_unchanged=True,
        task_transform='One-time settled-pose task initialization; both cues only; no online pose feed',
        task_transform_sha256=hashlib.sha256((REPO/'lewm/navigation_capability_target_reference_development.py').read_bytes()).hexdigest())
    with (root/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(output.dumps(result))


if __name__=='__main__':main()
