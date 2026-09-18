"""Full eighteen-fit admission and descriptive paired/optimization-seed readout."""
import argparse
import json
import time
import numpy as np
from scripts.augmented_family_switch_model_admission_development import admit
from scripts.run_go2_augmented_family_switch_fits_v1 import OUTPUT as FITS,ROSTER,OPTIMIZATION_SEEDS
from scripts.read_go2_augmented_family_switch_original_models_v1 import OUTPUT as ORIGINAL,FITS as OLD_FITS
from scripts.moving_action_switch_runtime_development import verify,hardware
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=BASE/'go2_augmented_family_switch_fit_readout_v1_attempt_001'
PROTOCOL='docs/go2_augmented_family_switch_fit_readout_v1_2026-09-08.md'
ORIGINAL_SHA='d312fc88fd13c5d0a9bb4543d6bead3e99f68cbf8f068af452a1e090c1da6c96'


def summarize_cells(cells):
    if not cells:raise ValueError('nonempty complete source/role/stratum cells required')
    motion=sum(c['motion_targets'] for c in cells);contact=sum(c['contact_targets'] for c in cells)
    undefined=sum(c['undefined_yaw'] for c in cells)
    return dict(motion_targets=motion,contact_targets=contact,contact_positives=sum(c['contact_positives'] for c in cells),
        undefined_yaw=undefined,
        position_error_m=sum(c['position_error_m']*c['motion_targets'] for c in cells if c['motion_targets'])/motion if motion else None,
        yaw_error_rad=sum(c['yaw_error_rad']*c['motion_targets'] for c in cells if c['motion_targets'])/motion if motion and not undefined else None,
        contact_brier=sum(c['contact_brier']*c['contact_targets'] for c in cells if c['contact_targets'])/contact if contact else None)


def metrics(scores):
    cells=scores['clusters'];keys=sorted({(c['source'],c['scope']) for c in cells})
    return {(source,scope):summarize_cells([c for c in cells if (c['source'],c['scope'])==(source,scope)]) for source,scope in keys}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit-result-sha256',required=True);args=parser.parse_args()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive complete fit readout; no retry/resume')
    admitted=admit(args.fit_result_sha256);study=read_json(FITS,'result.json');fit_launch=read_json(FITS,'launch.json')
    verify_artifacts(ORIGINAL,{'result.json':ORIGINAL_SHA});old=read_json(ORIGINAL,'result.json')
    if old['status']!='AUGMENTED_FAMILY_SWITCH_ORIGINAL_MODELS_COMPLETE':raise ValueError('complete original-model reference required')
    verify_artifacts(ORIGINAL,old['artifact_sha256']);reference=read_json(ORIGINAL,'launch.json');verify(reference)
    if reference['switch_input_result_sha256']!=study['science']['switch_input_result_sha256']:
        raise ValueError('old and augmented models must use the same branch inputs')
    verify_artifacts(OLD_FITS,reference['fit_artifact_sha256'])
    old_initial={read_json(OLD_FITS,name+'_fit.json')['fit']['initial_sha256'] for name in old['models']}
    new_initial={r['fit']['initial_sha256'] for r in study['records'] if r['fit']['seed']==2026091001}
    if len(old_initial)!=1 or old_initial!=new_initial:raise ValueError('first optimization seed must match original initialization')
    sources=dict(study['source_sha256'])
    for name,h in old['source_sha256'].items():
        if name in sources and sources[name]!=h:raise ValueError('reference and new fit source identities conflict')
        sources[name]=h
    sources=discover_sources((PROTOCOL,'scripts/read_go2_augmented_family_switch_fits_v1.py',
        'lewm/tests/test_augmented_family_switch_fit_readout_development.py'),sources)
    launch=fit_launch|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),hardware=hardware(),
        fit_result_sha256=args.fit_result_sha256,original_result_sha256=ORIGINAL_SHA,
        all_eighteen_admission=admitted,model_training=False,optimizer_steps=0,native_execution=False)
    if launch['hardware']['memory_available_bytes']<8*1024**3 or launch['hardware']['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('bounded readout RAM/storage allowance required')
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);start=time.perf_counter()
    try:
        baseline=read_json(ORIGINAL,'scores.json');rows=[];paired=[];groups={}
        for record in study['records']:
            name=record['name'];fit=record['fit']
            for role in ('train','geometry_transfer'):
                for (source,scope),value in metrics(read_json(FITS,name+'_'+role+'_scores.json')).items():
                    row=dict(model=name,seed=fit['seed'],variant=fit['input_variant'],condition=fit['condition'],
                        role=role,source=source,scope=scope,**value);rows.append(row)
                    groups.setdefault((fit['input_variant'],fit['condition'],role,source,scope),[]).append(row)
                    if fit['seed']==2026091001:
                        previous=metrics(baseline[name][role])[source,scope]
                        if any(value[k]!=previous[k] for k in ('motion_targets','contact_targets','contact_positives')):
                            raise ValueError('paired score target denominators changed')
                        delta={k:value[k]-previous[k] if value[k] is not None and previous[k] is not None else None
                            for k in ('position_error_m','yaw_error_rad','contact_brier')}
                        paired.append(dict(model=name,role=role,source=source,scope=scope,original=previous,augmented=value,
                            augmented_minus_original=delta,initial_model_exactly_matched=True))
        grouped=[]
        for (variant,condition,role,source,scope),members in sorted(groups.items()):
            if len(members)!=3 or {r['seed'] for r in members}!=set(OPTIMIZATION_SEEDS):raise ValueError('all three optimization seeds required')
            values={}
            for metric in ('position_error_m','yaw_error_rad','contact_brier'):
                selected=[r[metric] for r in members]
                values[metric]=dict(mean=float(np.mean(selected)),optimization_seed_sd=float(np.std(selected,ddof=1))) if all(v is not None for v in selected) else None
            grouped.append(dict(variant=variant,condition=condition,role=role,source=source,scope=scope,
                seeds=list(OPTIMIZATION_SEEDS),metrics=values,independent_maze_confidence_interval=False))
        write_json(OUTPUT/'metrics.json',dict(per_model=rows,first_seed_paired=paired,three_seed_descriptive=grouped))
        verify(launch);verify_artifacts(FITS,{'result.json':args.fit_result_sha256,**study['artifact_sha256']})
        verify_artifacts(ORIGINAL,{'result.json':ORIGINAL_SHA,**old['artifact_sha256']})
        result=dict(status='AUGMENTED_FAMILY_SWITCH_FIT_READOUT_COMPLETE',optimizer_updates_admitted=21600,
            final_models_admitted=18,first_seed_initialization_matches_original=True,
            primary_native_candidate=study['science']['primary_native_candidate'],checkpoint_selection_performed=False,
            optimization_seed_variation_is_not_independent_maze_replication=True,wall_s=time.perf_counter()-start,
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','metrics.json')},
            native_execution=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print('AUGMENTED_FAMILY_SWITCH_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AUGMENTED_FIT_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
