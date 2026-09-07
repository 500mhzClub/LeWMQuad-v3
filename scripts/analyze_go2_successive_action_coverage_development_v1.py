#!/usr/bin/env python3
"""Post-audit descriptive causal-coverage diagnostic; no fitting or new physics."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lewm.action_coverage_diagnostic_development import command_context,support_table,stratified_errors
from lewm.causal_subtrajectory_learning_development import AuditedSubtrajectoryDataset,DERIVATION_ROOT
from lewm.online_temporal_choice_development import STUDY,LAUNCH_SHA
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.run_go2_successive_choice_maze_development_v1 import OUTPUT as PHYSICAL,digest,verify,write_json

OUTPUT=ROOT/'.generated/go2_successive_action_coverage_development_v1_attempt_001'


def main():
    if len(sys.argv)!=1 or OUTPUT.exists(): raise ValueError('fixed fresh diagnostic output required')
    audit=json.loads((PHYSICAL/'raw_artifact_audit_clock_boundary_v2.json').read_text())
    witness=json.loads((PHYSICAL/'full_audit_source_dependency_witness_clock_boundary_v2.json').read_text())
    if (audit['status']!='PASS' or not audit['full_study'] or audit['audited_trials']!=144
            or audit['study_result_sha256']!=digest(PHYSICAL/'result.json')
            or witness['status']!='PASS' or witness['full_audit_sha256']!=digest(PHYSICAL/'raw_artifact_audit_clock_boundary_v2.json')):
        raise ValueError('complete physical audit and dependency witness required')
    verify(witness['source_sha256'])
    if digest(STUDY/'launch.json')!=LAUNCH_SHA: raise ValueError('fixed training identity')
    training_launch=json.loads((STUDY/'launch.json').read_text())
    verify(training_launch['source_sha256']|training_launch['input_sha256'])
    dataset=AuditedSubtrajectoryDataset(DERIVATION_ROOT,'train')
    sources={p:digest(ROOT/p) for p in ('lewm/action_coverage_diagnostic_development.py',
        'lewm/tests/test_action_coverage_diagnostic_development.py','scripts/analyze_go2_successive_action_coverage_development_v1.py')}
    inputs={str((PHYSICAL/name).relative_to(ROOT)):digest(PHYSICAL/name) for name in
        ('launch.json','result.json','raw_artifact_audit_clock_boundary_v2.json','full_audit_source_dependency_witness_clock_boundary_v2.json')}
    inputs[str((DERIVATION_ROOT/'windows.json').relative_to(ROOT))]=digest(DERIVATION_ROOT/'windows.json')
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',{'source_sha256':sources,'input_sha256':inputs,
        'scope':'exploratory post-panel action-history support; no causal effect claim, fitting, selection or commands'})
    training=[]; omitted=[]
    for row in dataset.rows:
        if not row['targets'][0]['contact_valid']:
            omitted.append(row['window_id']); continue
        packet=load_route_observation(dataset.corpus.paths[row['context_scene_id']],row['history_observation_indices'][-1])
        context=command_context(packet)
        training.append({'window_id':row['window_id'],'layout_id':row['layout_id'],
            'stage':'initial' if row['offset_ns']==0 else 'later',
            'prior_index':context['last_action_index'] if context['last_action_index'] is not None else 5,
            'future_index':row['action_index'],'constant_past':context['five_applied_ticks_constant']})
    table=support_table(training); physical=json.loads((PHYSICAL/'result.json').read_text()); online=[]
    for member in physical['trials']:
        directory=PHYSICAL/member['scene_id']
        if digest(directory/'result.json')!=member['result_sha256']: raise ValueError('trial binding changed')
        if digest(directory/'selection_events.json')!=member['artifact_sha256']['selection_events.json']: raise ValueError('choice binding changed')
        events=json.loads((directory/'selection_events.json').read_text()); previous=None
        for event,error in zip(events,member['metrics']['executed_decision_errors'],strict=True):
            choice=event['selection']; action=choice['selected_action_index']
            context=command_context(load_route_observation(directory,event['observation_index']))
            prior=context['last_action_index']
            support=next(r for r in table if r['stage']=='later' and r['prior_action_index']==prior and r['future_action_index']==action)
            group='initial' if previous is None else ('repeat_previous_selection' if previous==action else 'switch_previous_selection')
            label=error['label']; online.append({'scene_id':member['scene_id'],'layout_id':member['layout_id'],
                'method':member['method'],'intent':member['intent_name'],'decision_index':choice['decision_index'],
                'group':group,'prior_actual_context':context,'selected_action_index':action,
                'training_later_pair_windows':support['windows'],'training_later_pair_layouts':support['layouts'],
                'contact':bool(label['contact_by_horizon']) if label is not None and label['contact_valid'] else None,
                **{k:error[k] for k in ('position_error_m','yaw_error_rad','contact_brier')}})
            previous=action
    verify(sources|inputs)
    result={'status':'COMPLETE','training_windows':len(training),'omitted_training_unknown_contact':omitted,
        'training_layouts':len({r['layout_id'] for r in training}),'training_support':table,'online_choices':len(online),
        'online_rows':online,'stratified_errors':stratified_errors(online),'launch_sha256':digest(OUTPUT/'launch.json'),
        'limitations':['Pair support is not full scene/history support or counterfactual identifiability.',
            'Switch and continuation groups visit different states and have different censoring; differences are not causal switching effects.',
            'An initial teacher-stop context is not a moving-context counterfactual.',
            'No model selection, fitting, policy change or independent maze test occurs.']}
    write_json(OUTPUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('training_support','online_rows','stratified_errors')},indent=2))


if __name__=='__main__': main()
