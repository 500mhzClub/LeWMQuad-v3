"""Reconstruct all expanded-data coefficients before a named corrected reload."""
from lewm.all_phase_translation_bias_development import AllPhaseTranslationBiasModel
from scripts.all_phase_model_admission_development import admit as base_admit, load_assigned as base_load
from scripts.all_phase_study_inputs_development import stream
from scripts.fit_go2_all_phase_training_translation_bias_v1 import OUTPUT,FITS,reconstruct_coefficients
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.run_go2_successive_choice_maze_development_v1 import digest


def admit(result_sha256):
    verify_artifacts(OUTPUT,{'result.json':result_sha256});result=read_json(OUTPUT,'result.json')
    expected=dict(status='ALL_PHASE_TRAINING_TRANSLATION_BIAS_COMPLETE',models=18,trained_heads=30,
        fitted_scalars=480,training_contexts=4010,motionless_contexts=36,neural_weight_updates=0,
        optimizer_updates=0,corrected_transfer_evaluation=False,native_execution=False)
    if any(result[k]!=v for k,v in expected.items()):raise ValueError('complete training-only expanded correction required')
    ids=result['artifact_sha256']|{'result.json':result_sha256};verify_artifacts(OUTPUT,ids)
    launch=read_json(OUTPUT,'launch.json');verify_ordered_launch(launch)
    base=base_admit(result['fit_result_sha256'])
    if (launch['all_eighteen_admission']!=base or launch['source_sha256']!=result['source_sha256']
            or launch['fit_result_sha256']!=result['fit_result_sha256']):
        raise ValueError('unchanged exact full-fit admission/source identities required')
    data=stream(maximum_cache_bytes=0)
    expected_coefficients=reconstruct_coefficients(base,data.view,read_json(FITS,'training_schedules.json'))
    if read_json(OUTPUT,'coefficients.json')!=expected_coefficients:
        raise ValueError('every expanded training-only coefficient must reproduce exactly')
    complete=read_json(OUTPUT,'coefficients_complete.json')
    if (complete['coefficients_sha256']!=digest(OUTPUT/'coefficients.json') or complete['models']!=18
            or complete['trained_heads']!=30 or complete['fitted_scalars']!=480 or complete['corrected_transfer_evaluation']):
        raise ValueError('all coefficients frozen before corrected evaluation required')
    verify_ordered_launch(launch);verify_artifacts(OUTPUT,ids)
    return dict(correction_result_sha256=result_sha256,correction_artifact_sha256=ids,
        base_admission=base,coefficients=expected_coefficients,all_coefficients_reconstructed=True,
        all_models=18,all_trained_heads=30,navigation_qualified=False)


def load_assigned(admission,name):
    if (admission['all_coefficients_reconstructed'] is not True or admission['all_models']!=18
            or admission['all_trained_heads']!=30):
        raise ValueError('complete expanded correction admission required')
    verify_artifacts(OUTPUT,admission['correction_artifact_sha256'])
    coefficients=read_json(OUTPUT,'coefficients.json')
    if coefficients!=admission['coefficients'] or name not in coefficients:
        raise ValueError('exact assigned expanded correction required')
    base,condition,variant=base_load(admission['base_admission'],name);record=coefficients[name]
    if (record['condition'],record['variant'])!=(condition,variant):raise ValueError('assigned correction treatment changed')
    return AllPhaseTranslationBiasModel(base,record['heads']),condition,variant
