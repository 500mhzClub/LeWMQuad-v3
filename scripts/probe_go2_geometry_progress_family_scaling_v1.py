"""One fixed serial/four-process native throughput and identity comparison."""
import json
from lewm.geometry_progress_layout_family_development import assignments,APPEARANCES
from scripts.geometry_progress_family_runtime_development import preflight,run_phase,verify,hardware
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT=BASE/'go2_geometry_progress_family_scaling_v1_attempt_001'
SERIAL=BASE/'go2_geometry_progress_family_scaling_serial_v1_attempt_001'
PARALLEL=BASE/'go2_geometry_progress_family_scaling_parallel_v1_attempt_001'
PROTOCOL='docs/go2_geometry_progress_family_scaling_v1_2026-09-08.md'
COLLECTION_PROTOCOL='docs/go2_geometry_progress_family_collection_v1_2026-09-08.md'
SEEDS=(PROTOCOL,'scripts/probe_go2_geometry_progress_family_scaling_v1.py',
    'scripts/run_go2_geometry_progress_family_v1.py',COLLECTION_PROTOCOL,
    'lewm/geometry_progress_family_learning_sample_development.py',
    'lewm/tests/test_geometry_progress_layout_family_development.py',
    'lewm/tests/test_geometry_progress_family_execution_development.py',
    'lewm/tests/test_geometry_progress_family_learning_development.py',
    'lewm/tests/test_geometry_progress_family_accounting_development.py')


def cases():
    cells=assignments()
    return [next(t for t,c in cells.items() if c['geometry']==g and c['action']==a and c['appearance_seed']==APPEARANCES[0])
        for g,a in (('cluster_00_left_open','left_arc'),('cluster_00_right_open','forward'),
            ('cluster_01_left_open','hold'),('cluster_01_right_open','right_arc'))]


def decision(serial,parallel,comparisons):
    def admissible(phase):
        return phase['all_workers_completed'] and set(phase['records'])==set(cases()) and all(not r['hard_measurement_failed_frames']
            and (r['outcome']['complete_horizon'] or r['outcome']['physical_stop']=='DISALLOWED_CONTACT')
            and r['outcome']['acquisition_stop'] is None for r in phase['records'].values())
    if serial['wall_s']<=0 or parallel['wall_s']<=0:raise ValueError('measured positive phase times required')
    speedup=serial['wall_s']/parallel['wall_s']
    exact=bool(len(comparisons)==4 and {r['trial'] for r in comparisons}==set(cases()) and all(r['exact_equal'] for r in comparisons))
    serial_ok=admissible(serial);parallel_ok=admissible(parallel)
    return dict(serial_admissible=serial_ok,parallel_admissible=parallel_ok,all_execution_and_pixel_signatures_equal=exact,
        measured_speedup=speedup,minimum_speedup_for_four_workers=1.25,
        selected_workers=4 if serial_ok and parallel_ok and exact and speedup>=1.25 else 1 if serial_ok else None,
        training_data_reuse_permitted=False,qualification_granted=False)


def main():
    for root in (OUTPUT,SERIAL,PARALLEL):
        validate_root(root,must_exist=False)
        if root.exists() or root.is_symlink():raise ValueError('exclusive scaling bench; no retry/resume')
    trials=cases()
    launch=preflight(output=OUTPUT,protocol=PROTOCOL,seed_paths=SEEDS,planned_trials=trials,workers=4,storage_bytes=2*1024**3)
    launch.update(scope='infrastructure-only serial/four-process comparison; excluded from learning and transfer evaluation',
        serial_root=str(SERIAL),parallel_root=str(PARALLEL),native_episodes=8)
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        phases=[];phase_bindings={}
        for root,workers in ((SERIAL,1),(PARALLEL,4)):
            phase=launch|dict(output_root=str(root),native_scene_workers=workers,hardware=hardware(),parent_launch_sha256=digest(OUTPUT/'launch.json'))
            create_output(root);write_json(root/'launch.json',phase)
            run=run_phase(root,trials,workers=workers,record_signature=True);verify(phase)
            run['artifact_sha256']={n:h for r in run['records'].values() for n,h in r.get('artifact_sha256',{}).items()}
            for t in run['records']:
                for suffix in ('_worker.log','_worker_terminal.json'):
                    n=t+suffix
                    if (root/n).is_file():run['artifact_sha256'][n]=digest(root/n)
            for n in ('launch.json','resource_monitor.jsonl'):run['artifact_sha256'][n]=digest(root/n)
            verify_artifacts(root,run['artifact_sha256']);write_json(root/'result.json',run)
            phase_bindings[str(root)]={'result.json':digest(root/'result.json')};phases.append(run)
            if not run['all_workers_completed']:raise ValueError('native scaling phase infrastructure failure')
        comparisons=[]
        for trial in trials:
            a=read_json(SERIAL,trial+'_signature.json');b=read_json(PARALLEL,trial+'_signature.json')
            unequal=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
            comparisons.append(dict(trial=trial,signature_fields=len(a),exact_equal=not unequal,unequal_fields=unequal))
        result=decision(*phases,comparisons)
        verify(launch)
        for root,ids in phase_bindings.items():verify_artifacts(__import__('pathlib').Path(root),ids)
        result.update(status='GEOMETRY_PROGRESS_FAMILY_SCALING_COMPLETE',cases=trials,comparisons=comparisons,
            phase_result_sha256=phase_bindings,phase_wall_s=[p['wall_s'] for p in phases],
            source_sha256=launch['source_sha256'],launch_sha256=digest(OUTPUT/'launch.json'),
            model_trained=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print(json.dumps(result|{'source_sha256':len(result['source_sha256'])},indent=2),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FAMILY_SCALING_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
