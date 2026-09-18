"""Fit all training-only intercepts before a separate corrected readout."""
import time
import numpy as np
from scripts.observation_horizon_model_admission_development import admit
from scripts.run_go2_observation_horizon_fits_v1 import OUTPUT as FITS,ROSTER
from scripts.observation_horizon_fit_inputs_development import TARGETS
from lewm.training_translation_bias_development import fit_translation_bias
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_training_translation_bias_v1_attempt_001'
PROTOCOL='docs/go2_training_translation_bias_v1_2026-09-08.md'
FIT_SHA='45b4680b85c87bd69dcaed6a0058f091105632909661dba319d6c05f5b533418'
WINDOWS_SHA='c293e454ac391da377a282c61dc30dd6abbdd5274d253c963e4837a78e8f7811'


def load_arrays(name,role):
    if name not in ROSTER or role not in ('train','geometry_transfer'):raise ValueError('exact admitted model and role required')
    with np.load(FITS/(name+'_'+role+'.npz'),allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


def reconstruct_coefficients(study):
    verify_artifacts(TARGETS,{'windows.json':WINDOWS_SHA})
    rows=read_json(TARGETS,'windows.json');schedules=read_json(FITS,'training_schedules.json')
    coefficients={}
    for record in study['records']:
        name=record['name'];fit=record['fit'];arrays=load_arrays(name,'train')
        if len(arrays['indices'])!=408:raise ValueError('all 408 training contexts required')
        heads=('direct_outcomes',) if fit['condition']=='direct' else ('direct_outcomes','rollout_outcomes')
        coefficients[name]=dict(base_model_sha256=fit['model_sha256'],condition=fit['condition'],
            variant=fit['input_variant'],seed=fit['seed'],heads={head:fit_translation_bias(rows,arrays,
                schedules[str(fit['seed'])],head=head) for head in heads})
    if set(coefficients)!=set(ROSTER) or sum(len(r['heads']) for r in coefficients.values())!=30:
        raise ValueError('all eighteen models and thirty trained heads required')
    return coefficients


def main():
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive correction fit; no retry/resume')
    admission=admit(FIT_SHA);study=read_json(FITS,'result.json');old=read_json(FITS,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/fit_go2_training_translation_bias_v1.py',
        'scripts/training_translation_bias_model_admission_development.py',
        'lewm/tests/test_training_translation_bias_development.py',
        'docs/go2_observation_horizon_phase_and_bias_diagnosis_2026-09-08.md',
        'docs/go2_eight_step_planning_goal_probe_result_2026-09-08.md'),study['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('correction fitting resource allowance unavailable')
    verify_artifacts(TARGETS,{'windows.json':WINDOWS_SHA})
    launch=old|dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),hardware=resources,
        all_eighteen_admission=admission,fit_result_sha256=FIT_SHA,training_target_windows_sha256=WINDOWS_SHA,
        native_execution=False,neural_weight_updates=0,optimizer_updates=0,coefficient_fitting=True,
        corrected_transfer_evaluation=False,correction_protocol=PROTOCOL)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);started=time.perf_counter()
    print('TRAINING_TRANSLATION_BIAS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        coefficients=reconstruct_coefficients(study)
        write_json(OUTPUT/'coefficients.json',coefficients)
        write_json(OUTPUT/'coefficients_complete.json',dict(models=18,trained_heads=30,fitted_scalars=480,
            coefficients_sha256=digest(OUTPUT/'coefficients.json'),corrected_transfer_evaluation=False,
            primary_native_candidate='seed_2026091001_full_jepa',reference_native_candidate='seed_2026091001_full_direct'))
        verify(launch);verify_artifacts(FITS,{'result.json':FIT_SHA,**study['artifact_sha256']})
        verify_artifacts(TARGETS,{'windows.json':WINDOWS_SHA})
        write_json(OUTPUT/'result.json',dict(status='TRAINING_TRANSLATION_BIAS_FIT_COMPLETE',models=18,
            trained_heads=30,fitted_scalars=480,neural_weight_updates=0,optimizer_updates=0,
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','coefficients.json','coefficients_complete.json')},
            wall_s=time.perf_counter()-started,corrected_transfer_evaluation=False,native_execution=False,
            probability_calibrated=False,checkpoint_selection_performed=False,navigation_qualified=False,goal_achieved=False))
        print('TRAINING_TRANSLATION_BIAS_FIT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_TRANSLATION_BIAS_FIT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
