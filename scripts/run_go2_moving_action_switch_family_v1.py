"""One fixed 144-cell action-switch collection after measured concurrency."""
import argparse
import json
from pathlib import Path
from lewm.moving_action_switch_family_development import TRIALS
from lewm.moving_action_switch_accounting_development import summarize
from scripts.moving_action_switch_runtime_development import preflight,run_phase,verify
from scripts.probe_go2_moving_action_switch_scaling_v1 import OUTPUT as BENCH,SEEDS,COLLECTION_PROTOCOL as PROTOCOL
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT=BASE/'go2_moving_action_switch_family_v1_attempt_001'
STORAGE=24*1024**3


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scaling-result-sha256',required=True);args=parser.parse_args()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive new dataset; no retry/resume')
    ids={'result.json':args.scaling_result_sha256};verify_artifacts(BENCH,ids);bench=read_json(BENCH,'result.json')
    if (bench['status']!='MOVING_ACTION_SWITCH_SCALING_COMPLETE' or bench['selected_workers'] not in (1,4)
            or bench['conservative_scientific_estimate_bytes']>STORAGE):
        raise ValueError('admissible measured concurrency and conservative storage estimate required')
    ids['launch.json']=bench['launch_sha256'];verify_artifacts(BENCH,ids);verify(read_json(BENCH,'launch.json'))
    for root,bindings in bench['phase_result_sha256'].items():
        directory=Path(root);verify_artifacts(directory,bindings)
        verify_artifacts(directory,read_json(directory,'result.json')['artifact_sha256'])
    launch=preflight(output=OUTPUT,protocol=PROTOCOL,seed_paths=SEEDS,planned_trials=list(TRIALS),
        workers=bench['selected_workers'],storage_bytes=STORAGE)
    for name,h in bench['source_sha256'].items():
        if launch['source_sha256'].get(name)!=h:raise ValueError('collection must use exact benchmarked sources')
    launch.update(scaling_sha256=ids,scaling_decision={k:bench[k] for k in ('selected_workers','measured_speedup',
        'all_execution_and_pixel_signatures_equal','serial_admissible','parallel_admissible',
        'maximum_measured_episode_bytes','conservative_scientific_estimate_bytes')},
        data_scope='72 training and72 reused-cluster development geometry-transfer cells; zero independent mazes')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        phase=run_phase(OUTPUT,list(TRIALS),workers=bench['selected_workers'],record_signature=False)
        write_json(OUTPUT/'collection_terminal.json',phase)
        if not phase['all_workers_completed'] or not phase['all_measurement_gates_pass']:
            raise ValueError('collection infrastructure/measurement failure; remaining cells unlaunched; no training')
        reports=[read_json(OUTPUT,t+'_audit.json') for t in TRIALS]
        result=summarize(reports)
        result.update(status='MOVING_ACTION_SWITCH_COLLECTION_AND_AUDIT_COMPLETE',
            phase_wall_s=phase['wall_s'],workers=phase['workers'],
            artifact_sha256={n:h for row in phase['records'].values() for n,h in row['artifact_sha256'].items()},
            source_sha256=launch['source_sha256'],conditions={r['trial']:{k:r[k] for k in ('geometry','cluster','data_role',
                'prefix_action','suffix_action','outcome','frames','decisions','physics_samples','hard_measurement_failed_frames')} for r in reports})
        for n in ('launch.json','collection_terminal.json','resource_monitor.jsonl',
                *(t+s for t in TRIALS for s in ('_worker.log','_worker_terminal.json'))):
            result['artifact_sha256'][n]=digest(OUTPUT/n)
        verify(launch);verify_artifacts(OUTPUT,result['artifact_sha256']);verify_artifacts(BENCH,ids)
        write_json(OUTPUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_sha256','source_sha256','conditions','prefix_comparisons')},indent=2),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MOVING_ACTION_SWITCH_COLLECTION_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
