"""One complete 96-episode family collection, audit and role accounting."""
import argparse
import json
from pathlib import Path
from lewm.geometry_progress_layout_family_development import TRIALS
from lewm.geometry_progress_family_accounting_development import summarize
from scripts.geometry_progress_family_runtime_development import preflight,run_phase,verify
from scripts.geometry_progress_family_audit_development import compare_prefixes
from scripts.probe_go2_geometry_progress_family_scaling_v1 import OUTPUT as BENCH,SEEDS,COLLECTION_PROTOCOL as PROTOCOL
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT=BASE/'go2_geometry_progress_family_v1_attempt_001'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scaling-result-sha256',required=True);args=parser.parse_args()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive family dataset; no retry/resume')
    ids={'result.json':args.scaling_result_sha256};verify_artifacts(BENCH,ids);bench=read_json(BENCH,'result.json')
    if bench['status']!='GEOMETRY_PROGRESS_FAMILY_SCALING_COMPLETE' or bench['selected_workers'] not in (1,4):
        raise ValueError('complete admissible native concurrency comparison required')
    ids['launch.json']=bench['launch_sha256'];verify_artifacts(BENCH,ids);verify(read_json(BENCH,'launch.json'))
    for root,bindings in bench['phase_result_sha256'].items():verify_artifacts(Path(root),bindings)
    launch=preflight(output=OUTPUT,protocol=PROTOCOL,seed_paths=SEEDS,planned_trials=list(TRIALS),
        workers=bench['selected_workers'],storage_bytes=12*1024**3)
    for name,h in bench['source_sha256'].items():
        if launch['source_sha256'].get(name)!=h:raise ValueError('collection must use exact benchmarked sources')
    launch.update(scaling_sha256=ids,scaling_decision={k:bench[k] for k in ('selected_workers','measured_speedup',
        'all_execution_and_pixel_signatures_equal','serial_admissible','parallel_admissible')},
        data_scope='48 training and48 geometry-transfer episodes; zero independent complete-maze evaluations')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        phase=run_phase(OUTPUT,list(TRIALS),workers=bench['selected_workers'],record_signature=False)
        write_json(OUTPUT/'collection_terminal.json',phase)
        if not phase['all_workers_completed']:raise ValueError('family worker infrastructure failure; remaining episodes unlaunched')
        reports=[read_json(OUTPUT,t+'_audit.json') for t in TRIALS]
        result=summarize(reports)
        result.update(status='GEOMETRY_PROGRESS_FAMILY_COLLECTION_AND_AUDIT_COMPLETE',
            phase_wall_s=phase['wall_s'],workers=phase['workers'],prefix_comparisons=compare_prefixes(reports),
            artifact_sha256={n:h for row in phase['records'].values() for n,h in row['artifact_sha256'].items()},
            source_sha256=launch['source_sha256'],conditions={r['trial']:{k:r[k] for k in ('geometry','cluster','data_role',
                'appearance_seed','action','outcome','frames','decisions','physics_samples','hard_measurement_failed_frames')} for r in reports})
        for n in ('launch.json','collection_terminal.json','resource_monitor.jsonl',
                *(t+s for t in TRIALS for s in ('_worker.log','_worker_terminal.json'))):
            result['artifact_sha256'][n]=digest(OUTPUT/n)
        verify(launch);verify_artifacts(OUTPUT,result['artifact_sha256'])
        write_json(OUTPUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_sha256','source_sha256','conditions','prefix_comparisons')},indent=2),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FAMILY_COLLECTION_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
