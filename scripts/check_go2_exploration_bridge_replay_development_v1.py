#!/usr/bin/env python3
"""Fitted-checkpoint replay of the new bridge; not RGB place/exit evaluation."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.exploration_local_bridge_development import ExplorationLocalBridge
from lewm.memory.observed_exploration_development import ObservedExploration,PlaceFix,ExitObservation
from lewm.online_temporal_choice_development import OnlineTemporalChoice,METHODS
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.run_go2_successive_choice_maze_development_v1 import OUTPUT as PHYSICAL,digest,verify,write_json

OUTPUT=ROOT/'.generated/go2_exploration_bridge_replay_development_v1_attempt_001'
SOURCES=('lewm/memory/observed_exploration_development.py','lewm/tests/test_observed_exploration_development.py',
    'lewm/exploration_local_bridge_development.py','lewm/tests/test_exploration_local_bridge_development.py',
    'scripts/check_go2_exploration_bridge_replay_development_v1.py','lewm/tests/test_exploration_bridge_replay_development.py',
    'docs/go2_exploration_bridge_replay_development_v1_2026-09-05.md')


def compare_selection(actual,expected):
    if actual.keys()!=expected.keys(): raise ValueError('selection schema changed')
    maximum=0.
    for key in actual.keys()-{'adapter_ms','inference_ms'}:
        if key in ('initial_direction_xy','direction_current_body_xy','candidate_costs'):
            if actual[key] is None or expected[key] is None:
                if actual[key]!=expected[key]: raise ValueError('missing numeric selection field')
            else:
                a,b=np.asarray(actual[key]),np.asarray(expected[key])
                if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('selection numeric shape or finiteness')
                error=float(np.max(np.abs(a-b))); maximum=max(maximum,error)
                if error>1e-12: raise ValueError('bearing reconstruction roundoff exceeded')
        elif actual[key]!=expected[key]: raise ValueError('exact bridge selection mismatch: '+key)
    return maximum


def main():
    if len(sys.argv)!=1 or OUTPUT.exists(): raise ValueError('fixed fresh bridge replay required')
    inputs={str((PHYSICAL/name).relative_to(ROOT)):sha for name,sha in {
        'launch.json':'e9a4bd281e631f06e01a134c3d68b969613d4599bf0554299a55d34bc20f7bf5',
        'result.json':'bd97ff363da74beebcb4a4770e9182b6554c791167a5a360d8535155d2bf8c7f',
        'raw_artifact_audit_clock_boundary_v2.json':'659c7d552ef49e0e9328a341da60fb97f2160639c33ba8fc7495783d4dba6d1f',
        'full_audit_source_dependency_witness_clock_boundary_v2.json':'f91c7537ad8897fdf835545a05fe557d73acc56c4fe568ad8da8c73767e28398'}.items()}
    verify(inputs); launch=json.loads((PHYSICAL/'launch.json').read_text())
    witness=json.loads((PHYSICAL/'full_audit_source_dependency_witness_clock_boundary_v2.json').read_text())
    sources=launch['source_sha256']|witness['source_sha256']|{p:digest(ROOT/p) for p in SOURCES}; verify(sources)
    report=json.loads((PHYSICAL/'result.json').read_text())
    if report['status']!='COMPLETE' or report['completed_trials']!=144: raise ValueError('full source panel required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    templates={m:OnlineTemporalChoice.from_completed_study(m) for m in METHODS}
    if {m:t.bindings for m,t in templates.items()}!=launch['model_bindings']: raise ValueError('fixed source model ensembles required')
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',{'source_sha256':sources,'input_sha256':inputs,
        'model_bindings':{m:t.bindings for m,t in templates.items()},
        'scope':'same actual packets and fitted checkpoints; symbolic exit bearing supplied from old directional task, not a perception or maze result'})
    rows=[]; maximum=0.
    try:
        for member in report['trials']:
            directory=PHYSICAL/member['scene_id']
            if digest(directory/'result.json')!=member['result_sha256']: raise ValueError('source member identity')
            names=['selection_events.json','observed_indices.json','policy_histories.npz','policy_observations.json']
            names.extend(f'rgb_{i:04d}.png' for i in range(member['rgb_packets']))
            verify({str((directory/name).relative_to(ROOT)):member['artifact_sha256'][name] for name in names})
            events=json.loads((directory/'selection_events.json').read_text()); by_packet={e['observation_index']:e for e in events}
            observed=json.loads((directory/'observed_indices.json').read_text()); initial=member['branch_start_observation_index']
            if initial<3 or not set(range(initial-3,initial+1))<=set(observed): raise ValueError('four actual initial packets required')
            memory=ObservedExploration(); bridge=ExplorationLocalBridge(memory,templates[member['method']],(0,0,0)); choices=0
            for index in (i for i in observed if i>=initial-3):
                packet=load_route_observation(directory,index); ns=packet['image']['measured_ns']; bridge.observe(packet,now_ns=ns)
                if index==initial:
                    memory.observe_place(PlaceFix('synthetic-conditioning-fix',ns,'conditioning-place',True))
                    x,y=member['intent_xy_body_start_m']
                    memory.observe_exit(ExitObservation('supplied-direction-cue',ns,'conditioning-place','symbolic-exit',math.atan2(y,x)))
                    selection=bridge.start_next('unqualified-symbolic-attempt',now_ns=ns)['local_selection']
                elif index in by_packet: selection=bridge.select_active(now_ns=ns)
                else: continue
                if index not in by_packet: raise ValueError('bridge invented source decision')
                maximum=max(maximum,compare_selection(selection,by_packet[index]['selection'])); choices+=1
            if choices!=len(events): raise ValueError('source decision omitted')
            if memory.attempts or memory.graph.places!={'conditioning-place'} or memory.pending is None:
                raise ValueError('replay invented an observed destination or completed traversal')
            rows.append({'scene_id':member['scene_id'],'method':member['method'],'replayed_choices':choices,
                'qualified_traversals':0,'input_and_model_predictions_exact':True})
            print(json.dumps({'event':'bridge_stream_replayed','completed':len(rows),'planned':144,'choices':choices}),flush=True)
        verify(sources|inputs)
        result={'status':'PASS','replayed_streams':len(rows),'replayed_choices':sum(r['replayed_choices'] for r in rows),
            'maximum_direction_or_cost_roundoff':maximum,'qualified_traversals':0,'trials':rows,
            'launch_sha256':digest(OUTPUT/'launch.json'),
            'scope':'integration replay only; no RGB place/exit detector, physical execution, fitting, beacon discovery or navigation claim'}
        write_json(OUTPUT/'result.json',result); print(json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',{'status':'FAIL','error':str(error),'replayed_streams':len(rows),'trials':rows,
            'launch_sha256':digest(OUTPUT/'launch.json')}); raise


if __name__=='__main__': main()
