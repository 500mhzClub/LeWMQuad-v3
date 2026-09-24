"""Exact inventory/batch bindings and explicit episode artifact roster."""
import json
from lewm.independent_layout_collection_development import CollectionInventory
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.navigation_artifact_root_development import BASE,validate_root
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings

INVENTORY_ROOT=ROOT/'.generated/go2_independent_layout_inventory_v1_attempt_001'
INVENTORY_IDS={
    'launch.json':'6497bad46510ebf49f95c2907b7ae992b937b667374f43da0ffad977742894db',
    'inventory.json':'714161041c6db96270d91b53749a982542a5b32fbbee155ec0d5ab98d7afd426',
    'structural_audit.json':'5107bda784a4c37b98f42976f5d9f8971ad6c68a3f9a363cdb329bacd4c2059d',
    'result.json':'525382c0e14819fbc5c9b1633641e9f9b46210e03831cf450b2e20904b5c34f1',
}
BATCHES=tuple(f'l{i:02d}' for i in range(12))
PROTOCOL='docs/go2_independent_layout_collection_v1_2026-09-06.md'
RESERVE=40*1024**3
BATCH_BUDGET=8*1024**3
EPISODE_ALLOWANCE=256*1024**2


def inventory_bindings():return {str((INVENTORY_ROOT/n).relative_to(ROOT)):h for n,h in INVENTORY_IDS.items()}


def load_inventory():
    verify_bindings(inventory_bindings())
    inventory=CollectionInventory(read_json(INVENTORY_ROOT,'inventory.json'))
    verify_bindings(inventory_bindings());return inventory


def output_root(batch):
    if batch not in BATCHES:raise ValueError('exact declared layout batch required')
    output=BASE/f'go2_independent_layout_collection_v1_{batch}_attempt_001'
    validate_root(output,must_exist=output.exists());return output


def validate_launch(launch,inventory,batch):
    if launch['batch']!=batch or launch['output_root']!=str(output_root(batch)):
        raise ValueError('bound batch/output identity required')
    ids=inventory.episode_ids(batch)
    if launch['planned_trials']!=list(ids):raise ValueError('complete prospective batch membership required')
    expected={c:inventory.specification(c) for c in ids}
    if json.dumps(launch['conditions'],sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False):
        raise ValueError('exact frozen episode constructions required')
    if (launch['maximum_batch_bytes']!=BATCH_BUDGET or launch['episode_storage_allowance_bytes']!=EPISODE_ALLOWANCE
            or launch['minimum_free_bytes']!=RESERVE or launch['inventory_sha256']!=INVENTORY_IDS['inventory.json']):
        raise ValueError('frozen inventory and resource limits required')


def episode_artifacts(spec,result):
    names=['specification.json','actuator_identity.json','floor_roles.json','terminal_actuator_gains.json',
        'terminal_native_robot_geometry.json','terminal_environment_identity.json','result.json',
        'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json',
        'ideal_sensor_samples.npz','policy_histories.npz','policy_observations.json','camera_audit.json',
        'depth_observations.json','depth_camera_audit.json','fast_gyro_samples.npz','fast_gyro_histories.npz',
        'floor_visual_collision_identity.json','context_decisions.json','command_tape.json','native_guard_rows.json','friction_checks.json',
        'visual_meshes/ground_visual.ply']
    if result['setup_checked']:names+=['static_objects.json','startup_native_robot_geometry.json','setup_checks.json']
    names+=['visual_meshes/'+b['wall_id']+'_visual.ply' for b in spec['geometry']['wall_boxes']]
    return names+[f'{prefix}_{i:04d}.{suffix}' for i in range(result['rgbd_frames'])
        for prefix,suffix in (('rgb','png'),('depth','npz'),('native_depth','npz'))]


def commit_episode(output,spec,result):
    names=[spec['trial']+'/'+p for p in episode_artifacts(spec,result)]
    present=[p for p in names if (output/p).is_file()]
    return dict(trial=spec['trial'],result=result,artifact_sha256={p:digest(output/p) for p in present},
        absent_expected_artifacts=sorted(set(names)-set(present)),
        artifact_bytes=sum((output/p).stat().st_size for p in present))
