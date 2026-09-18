"""Fixed serial/four-worker comparison of complete moving-prefix branches."""
import json
from pathlib import Path
from lewm.moving_action_switch_family_development import assignments,CANONICAL
from scripts.moving_action_switch_runtime_development import preflight,run_phase,verify,hardware,admissible_record
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT=BASE/'go2_moving_action_switch_scaling_v1_attempt_001'
SERIAL=BASE/'go2_moving_action_switch_scaling_serial_v1_attempt_001'
PARALLEL=BASE/'go2_moving_action_switch_scaling_four_v1_attempt_001'
PROTOCOL='docs/go2_moving_action_switch_scaling_v1_2026-09-08.md'
COLLECTION_PROTOCOL='docs/go2_moving_action_switch_family_collection_v1_2026-09-08.md'
SEEDS=(PROTOCOL,COLLECTION_PROTOCOL,'docs/go2_moving_action_switch_family_design_v1_2026-09-08.md',
    'scripts/probe_go2_moving_action_switch_scaling_v1.py','scripts/run_go2_moving_action_switch_family_v1.py',
    'lewm/tests/test_moving_action_switch_family_development.py',
    'lewm/tests/test_moving_action_switch_accounting_development.py',
    'lewm/tests/test_moving_action_switch_scaling_development.py')


def cases():
    cells=assignments()
    return [next(t for t,c in cells.items() if c['cluster']==cluster and c['prefix_action']=='left_arc'
        and c['suffix_action']=='hold') for cluster in CANONICAL]


def admissible_phase(phase):
    return bool(phase['all_workers_completed'] and phase['all_measurement_gates_pass']
        and set(phase['records'])==set(cases()) and all(admissible_record(r)
            and r['outcome']['complete_schedule'] and r['collection']['command_ticks']==63
            and r['collection']['rgbd_frames']==64 and r['collection']['physics_samples']==3900
            for r in phase['records'].values()))


def decision(serial,parallel,comparisons):
    if serial['wall_s']<=0 or parallel['wall_s']<=0:raise ValueError('measured positive phase times required')
    speedup=serial['wall_s']/parallel['wall_s']
    exact=bool(len(comparisons)==4 and {r['trial'] for r in comparisons}==set(cases()) and all(r['exact_equal'] for r in comparisons))
    serial_ok=admissible_phase(serial);parallel_ok=admissible_phase(parallel)
    return dict(serial_admissible=serial_ok,parallel_admissible=parallel_ok,
        all_execution_and_pixel_signatures_equal=exact,measured_speedup=speedup,
        minimum_speedup_for_four_workers=1.25,
        selected_workers=4 if serial_ok and parallel_ok and exact and speedup>=1.25 else 1 if serial_ok else None,
        training_data_reuse_permitted=False,qualification_granted=False)


def main():
    for root in (OUTPUT,SERIAL,PARALLEL):
        validate_root(root,must_exist=False)
        if root.exists() or root.is_symlink():raise ValueError('exclusive scaling comparison; no retry/resume')
    trials=cases()
    launch=preflight(output=OUTPUT,protocol=PROTOCOL,seed_paths=SEEDS,planned_trials=trials,workers=4,storage_bytes=2*1024**3)
    launch.update(scope='infrastructure-only; no benchmark outputs in learning or transfer evaluation',
        serial_root=str(SERIAL),parallel_root=str(PARALLEL),native_episodes=8)
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        phases=[];phase_bindings={};maximum_episode_bytes=0
        for root,workers in ((SERIAL,1),(PARALLEL,4)):
            phase=launch|dict(output_root=str(root),native_scene_workers=workers,hardware=hardware(),parent_launch_sha256=digest(OUTPUT/'launch.json'))
            create_output(root);write_json(root/'launch.json',phase)
            run=run_phase(root,trials,workers=workers,record_signature=True);verify(phase)
            run['artifact_sha256']={n:h for r in run['records'].values() for n,h in r.get('artifact_sha256',{}).items()}
            for t,row in run['records'].items():
                episode_bytes=sum((root/n).stat().st_size for n in row.get('artifact_sha256',{}))
                maximum_episode_bytes=max(maximum_episode_bytes,episode_bytes)
                row['artifact_bytes']=episode_bytes
                for suffix in ('_worker.log','_worker_terminal.json'):
                    n=t+suffix
                    if (root/n).is_file():run['artifact_sha256'][n]=digest(root/n)
            for n in ('launch.json','resource_monitor.jsonl'):run['artifact_sha256'][n]=digest(root/n)
            verify_artifacts(root,run['artifact_sha256']);write_json(root/'result.json',run)
            phase_bindings[str(root)]={'result.json':digest(root/'result.json')};phases.append(run)
            if not admissible_phase(run):raise ValueError('scaling phase failed completeness or measurement gate; no further phase')
        comparisons=[]
        for trial in trials:
            a=read_json(SERIAL,trial+'_signature.json');b=read_json(PARALLEL,trial+'_signature.json')
            unequal=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
            comparisons.append(dict(trial=trial,signature_fields=len(a),exact_equal=not unequal,unequal_fields=unequal))
        result=decision(*phases,comparisons)
        verify(launch)
        for root,ids in phase_bindings.items():verify_artifacts(Path(root),ids)
        result.update(status='MOVING_ACTION_SWITCH_SCALING_COMPLETE',cases=trials,comparisons=comparisons,
            phase_result_sha256=phase_bindings,phase_wall_s=[p['wall_s'] for p in phases],
            maximum_measured_episode_bytes=maximum_episode_bytes,
            conservative_scientific_estimate_bytes=maximum_episode_bytes*180+512*1024**2,
            source_sha256=launch['source_sha256'],launch_sha256=digest(OUTPUT/'launch.json'),
            model_trained=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print(json.dumps(result|{'source_sha256':len(result['source_sha256'])},indent=2),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MOVING_ACTION_SWITCH_SCALING_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
