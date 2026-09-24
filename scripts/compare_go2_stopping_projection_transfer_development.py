"""Compare the three completed controller assignments on one fresh maze."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.compare_go2_post_training_transfer_development import SHARED_FIELDS
from scripts.compare_go2_combined_perception_motion_development import behavior
from scripts.run_go2_stopping_projection_transfer_development import CONDITIONS, ROOT, INVENTORY_SHA256

FIELDS=SHARED_FIELDS+('local_view_reference_bank','maximum_extra_local_view_references',
    'recent_reference_refresh_from_accepted_anchor','maximum_recent_reference_age_ns',
    'committed_camera_view_turn','floor_reacquisition_enabled','planned_native_assignments',
    'planned_layout_indices','fixed_dispatch_pairs')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layout-index',type=int,choices=range(4),required=True)
    index=parser.parse_args().layout_index
    output=path(f'go2_stopping_projection_transfer_comparison_layout{index:02d}_v1_attempt_001')
    if output.exists():raise ValueError('preserve completed comparison')
    summaries={};launches={};sources={};treatments={};behaviors={};views={};corrections={}
    for condition in CONDITIONS:
        root=path(ROOT.format(index=index,condition=condition));launch=read(root,'launch.json')
        if (root/'launch_annotation_correction.json').exists():
            correction=read(root,'launch_annotation_correction.json')
            if hashlib.sha256((root/'launch.json').read_bytes()).hexdigest()!=correction['original_launch_sha256']:
                raise ValueError('annotation correction no longer binds original launch')
            launch=launch|correction['corrected_fields'];corrections[condition]=correction
        if (launch['comparison_condition']!=condition or launch['layout_index']!=index
                or launch['frozen_layout_inventory_sha256']!=INVENTORY_SHA256
                or launch['planned_native_assignments']!=12):
            raise ValueError('fixed transfer assignment differs')
        launches[condition]=launch;sources[condition]=launch['source_sha256']|launch['extra_sources']
        treatments[condition]=read(root,'actual_controller_treatment_v1.json')
        if treatments[condition]['condition']!=condition:raise ValueError('actual treatment record differs')
        summaries[condition]=read(root,'live_navigation_summary_v1.json')
        plans=[p for p in read(root,'planning.json') if 'selection' in p]
        if len(plans)!=treatments[condition]['selected_plans']:raise ValueError('evaluated plan population differs')
        behaviors[condition]=behavior(root,plans)
        behaviors[condition]['stopping_projection_changes']=sum(bool(
            p['selection'].get('planned_stopping_projection',{}).get('changed')) for p in plans)
        events=read(root,'frontier_visits.json')['events']
        views[condition]=dict(completed_events=len(events),
            completion_reasons=dict(Counter(e.get('completion_reason') for e in events)),
            maximum_completed_event_s=max([(e['completed_ns']-e['started_ns'])/1e9 for e in events],default=0))
    reference=launches['learned']
    differences={c:[k for k in FIELDS if v.get(k)!=reference.get(k)] for c,v in launches.items()}
    if any(differences.values()):raise ValueError(f'shared settings differ: {differences}')
    common=set.intersection(*(set(v) for v in sources.values()))
    exception=json.loads(Path('.generated/stopping_projection_transfer_annotation_fix_2026-09-15/correction.json').read_text())
    allowed={exception['original_sha256'],exception['corrected_sha256']};exceptions={};equal={}
    for name in sorted(common):
        hashes={c:v[name] for c,v in sources.items()}
        if len(set(hashes.values()))==1:equal[name]=next(iter(hashes.values()));continue
        if (Path(name).name!='run_go2_stopping_projection_transfer_development.py'
                or not set(hashes.values())<=allowed or not exception['runtime_classes_unchanged']
                or not exception['main_unchanged_except_writer_binding']):
            raise ValueError(f'unexpected common-source change: {name}')
        exceptions[name]=hashes
    report=dict(layout_index=index,conditions=summaries,actual_treatments=treatments,
        matched_settings={k:reference.get(k) for k in FIELDS},common_sources=equal,
        annotation_only_source_exception=exceptions,launch_annotation_corrections=corrections,
        behavior_metrics=behaviors,view_metrics=views,
        comparison='learned_and_fitted_predictive_motion_vs_instantaneous_reactive',
        fresh_development_layout=True,prior_registry_layout_count=72,
        same_maze_family=True,reactive_prediction_and_recovery_rules_differ=True,
        isolated_predictive_ranking_effect_established=False,jepa_training_effect_established=False,
        statistical_advantage_established=False,host_real_time_qualified=False,hardware_validated=False)
    output.mkdir()
    with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({c:dict(round_trip=s['independent_arrival_evaluation']['round_trip_arrival_checks_passed'],
        contacts=s['independent_arrival_evaluation']['disallowed_contact_samples']) for c,s in summaries.items()}))


if __name__=='__main__':main()
