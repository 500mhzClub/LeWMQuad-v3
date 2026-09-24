"""Reconstruct every training-only intercept before assigned native reloads."""
from lewm.training_translation_bias_development import TrainingTranslationBiasModel
from scripts.observation_horizon_model_admission_development import admit as base_admit,load_assigned as base_load
from scripts.fit_go2_training_translation_bias_v1 import OUTPUT,FITS,FIT_SHA,reconstruct_coefficients
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify


def admit(result_sha256):
    verify_artifacts(OUTPUT,{'result.json':result_sha256});result=read_json(OUTPUT,'result.json')
    if (result['status']!='TRAINING_TRANSLATION_BIAS_FIT_COMPLETE' or result['models']!=18
            or result['trained_heads']!=30 or result['fitted_scalars']!=480
            or result['neural_weight_updates']!=0 or result['optimizer_updates']!=0
            or result['corrected_transfer_evaluation'] or result['native_execution']):
        raise ValueError('complete training-only thirty-head correction required')
    ids={'result.json':result_sha256,**result['artifact_sha256']};verify_artifacts(OUTPUT,ids)
    launch=read_json(OUTPUT,'launch.json');verify(launch)
    base=base_admit(FIT_SHA)
    if launch['all_eighteen_admission']!=base or launch['source_sha256']!=result['source_sha256']:
        raise ValueError('unchanged original eighteen-model admission and source required')
    study=read_json(FITS,'result.json');expected=reconstruct_coefficients(study)
    if read_json(OUTPUT,'coefficients.json')!=expected:raise ValueError('every training-only coefficient must reproduce exactly')
    complete=read_json(OUTPUT,'coefficients_complete.json')
    if (complete['coefficients_sha256']!=digest(OUTPUT/'coefficients.json') or complete['models']!=18
            or complete['trained_heads']!=30 or complete['fitted_scalars']!=480 or complete['corrected_transfer_evaluation']):
        raise ValueError('all coefficients must be frozen before corrected transfer evaluation')
    verify(launch);verify_artifacts(OUTPUT,ids)
    return dict(correction_result_sha256=result_sha256,correction_artifact_sha256=ids,
        base_admission=base,coefficients=expected,all_coefficients_reconstructed=True,
        all_models=18,all_trained_heads=30,navigation_qualified=False)


def load_assigned(admission,name):
    if not admission['all_coefficients_reconstructed'] or admission['all_models']!=18 or admission['all_trained_heads']!=30:
        raise ValueError('complete correction admission required')
    verify_artifacts(OUTPUT,admission['correction_artifact_sha256'])
    coefficients=read_json(OUTPUT,'coefficients.json')
    if coefficients!=admission['coefficients'] or name not in coefficients:raise ValueError('exact assigned correction required')
    base,condition,variant=base_load(admission['base_admission'],name);record=coefficients[name]
    if (record['condition'],record['variant'])!=(condition,variant):raise ValueError('assigned correction treatment changed')
    return TrainingTranslationBiasModel(base,record['heads']),condition,variant
