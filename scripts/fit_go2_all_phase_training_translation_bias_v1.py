"""Derive all expanded-data training intercepts before corrected evaluation."""
import argparse
import time
import numpy as np
from lewm.all_phase_translation_bias_development import fit_translation_bias
from lewm.all_phase_training_schedule_development import schedule as build_schedule, SEEDS
from scripts.all_phase_model_admission_development import admit
from scripts.all_phase_study_inputs_development import stream, corrected_inputs
from scripts.all_phase_fit_execution_development import OUTPUT as FITS, ROSTER, RESERVE
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_all_phase_training_translation_bias_v1_attempt_001'
PROTOCOL = 'docs/go2_all_phase_training_translation_bias_v1_2026-09-10.md'
SOURCE = 'scripts/fit_go2_all_phase_training_translation_bias_v1.py'
ALLOWANCE = 256*1024**2


def reconstruct_coefficients(admission, view, schedules):
    expected_names = {r['name'] for r in ROSTER}
    if (set(admission['snapshots'])!=expected_names
            or admission['all_eighteen_ledgers_raw_scores_and_snapshots_reconstructed'] is not True
            or len(view.indices('train'))!=4010):
        raise ValueError('complete admitted eighteen-model expanded training population required')
    if schedules!={str(seed):build_schedule(view,seed=seed) for seed in SEEDS}:
        raise ValueError('all exact original expanded-context schedules required')
    coefficients = {}
    for assignment in ROSTER:
        name = assignment['name']; snapshot = admission['snapshots'][name]
        prefix = snapshot['filename'][:-3]
        filename = prefix+'_train.npz'
        verify_artifacts(FITS,{filename:admission['fit_artifact_sha256'][filename]})
        with np.load(FITS/filename,allow_pickle=False) as saved: arrays={k:saved[k] for k in saved.files}
        if arrays['indices'].tolist()!=view.indices('train'):
            raise ValueError('all and only4010 training prediction contexts required')
        condition,variant,seed=(assignment[k] for k in ('condition','variant','seed'))
        heads=('direct_outcomes',) if condition=='direct' else ('direct_outcomes','rollout_outcomes')
        records={head:fit_translation_bias(view.rows,arrays,schedules[str(seed)],head=head) for head in heads}
        if any(r['training_examples']!=4010 or r['training_draws']!=7200
                or r['motionless_training_examples']!=36 for r in records.values()):
            raise ValueError('complete available and motion-censored training accounting required')
        coefficients[name]=dict(base_model_sha256=snapshot['model_sha256'],condition=condition,variant=variant,
            seed=seed,schedule_sha256=schedules[str(seed)]['schedule_sha256'],heads=records)
    if len(coefficients)!=18 or sum(len(r['heads']) for r in coefficients.values())!=30:
        raise ValueError('all eighteen assigned models and thirty trained heads required')
    return coefficients


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit-result-sha256',required=True);args=parser.parse_args()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive correction; no retry/resume')
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<RESERVE+ALLOWANCE:
        raise ValueError('training-only correction resource allowance unavailable')
    admission=admit(args.fit_result_sha256);study=read_json(FITS,'result.json');old=read_json(FITS,'launch.json')
    data=stream(maximum_cache_bytes=0);schedules=read_json(FITS,'training_schedules.json')
    sources=discover_sources((SOURCE,PROTOCOL,'scripts/all_phase_translation_bias_model_admission_development.py',
        'lewm/tests/test_all_phase_translation_bias_development.py',
        'lewm/tests/test_all_phase_model_admission_development.py'),study['source_sha256'])
    keys=('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules')
    launch={k:old[k] for k in keys}
    launch.update(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),hardware=resources,
        all_eighteen_admission=admission,fit_result_sha256=args.fit_result_sha256,
        input_identity=old['science']['input_identity'],neural_weight_updates=0,optimizer_updates=0,
        coefficient_fitting=True,private_training_future_materialization=False,
        geometry_transfer_future_materialization=False,future_rgb_materialization=False,
        native_training_targets_used=True,transfer_targets_used_for_coefficients=False,
        corrected_transfer_evaluation=False,native_execution=False,navigation_qualified=False)
    verify_ordered_launch(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    started=time.perf_counter();print('ALL_PHASE_TRANSLATION_BIAS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        coefficients=reconstruct_coefficients(admission,data.view,schedules)
        write_json(OUTPUT/'coefficients.json',coefficients)
        write_json(OUTPUT/'coefficients_complete.json',dict(models=18,trained_heads=30,fitted_scalars=480,
            coefficients_sha256=digest(OUTPUT/'coefficients.json'),corrected_transfer_evaluation=False,
            primary_native_candidate=old['science']['primary_native_candidate']))
        corrected_inputs();verify_ordered_launch(launch)
        verify_artifacts(FITS,study['artifact_sha256']|{'result.json':args.fit_result_sha256})
        names=('launch.json','coefficients.json','coefficients_complete.json')
        if sum((OUTPUT/n).stat().st_size for n in names)>ALLOWANCE:raise ValueError('correction output allowance exceeded')
        write_json(OUTPUT/'result.json',dict(status='ALL_PHASE_TRAINING_TRANSLATION_BIAS_COMPLETE',
            fit_result_sha256=args.fit_result_sha256,source_sha256=sources,
            artifact_sha256={n:digest(OUTPUT/n) for n in names},models=18,trained_heads=30,fitted_scalars=480,
            training_contexts=4010,motionless_contexts=36,neural_weight_updates=0,optimizer_updates=0,
            wall_s=time.perf_counter()-started,corrected_transfer_evaluation=False,native_execution=False,
            probability_calibrated=False,checkpoint_selection_performed=False,navigation_qualified=False))
        print('ALL_PHASE_TRANSLATION_BIAS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ALL_PHASE_TRANSLATION_BIAS_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
