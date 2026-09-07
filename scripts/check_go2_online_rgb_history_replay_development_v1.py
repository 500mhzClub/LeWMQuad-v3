#!/usr/bin/env python3
"""Replay exact audited development RGB streams through the live history buffer."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lewm.causal_subtrajectory_learning_development import AuditedSubtrajectoryDataset,DERIVATION_ROOT,causal_history_tensors
from lewm.online_rgb_history_development import OnlineRGBHistory
from lewm.route_rgb_dataset_development import load_route_observation


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def main():
    output=ROOT/'.generated/go2_online_rgb_history_replay_development_v1_attempt_001'
    if output.exists(): raise ValueError('fresh exact stream-replay output required')
    dataset=AuditedSubtrajectoryDataset(DERIVATION_ROOT,'train')
    windows=json.loads((DERIVATION_ROOT/'windows.json').read_text())
    sources=('lewm/online_rgb_history_development.py','lewm/tests/test_online_rgb_history_development.py',
        'scripts/check_go2_online_rgb_history_replay_development_v1.py')
    bindings={name:sha(ROOT/name) for name in sources}
    output.mkdir(); write(output/'launch.json',{'source_sha256':bindings,'windows_sha256':sha(DERIVATION_ROOT/'windows.json'),
        'tensor_check_sha256':sha(DERIVATION_ROOT/'tensor_interface_check.json'),'scope':'recorded policy-stream replay; no fitting or physical commands'})
    rows=[]; counts={'packets_pushed':0,'ready_packets':0,'derived_windows_compared':0,
        'noncanonical_initial_windows_excluded':0,'non_command_clock_terminal_frames':0}
    try:
        for scene,directory in dataset.corpus.paths.items():
            manifest=json.loads((directory/'policy_observations.json').read_text())
            targets={w['decision_ns']:w for w in windows if w['scene_id']==scene}
            history=OnlineRGBHistory(); history.begin_episode((0,0,0)); cache={}; comparisons=0
            for index,frame in enumerate(manifest['frames']):
                packet=load_route_observation(directory,index); ns=frame['decision_ns']
                if ns%100_000_000:
                    if index!=len(manifest['frames'])-1: raise ValueError('nonterminal off-clock frame')
                    counts['non_command_clock_terminal_frames']+=1; continue
                cache[index]=packet; status=history.push(packet,now_ns=ns); counts['packets_pushed']+=1
                counts['ready_packets']+=status['ready']
                if ns not in targets: continue
                window=targets[ns]
                if window['context_scene_id']!=scene:
                    if window['offset_ns']!=0: raise ValueError('noninitial borrowed context')
                    counts['noncanonical_initial_windows_excluded']+=1; continue
                actual=history.tensors(now_ns=ns)
                expected=causal_history_tensors([cache[i] for i in window['history_observation_indices']],ns)
                if any(not torch.equal(actual[k],expected[k]) for k in actual): raise ValueError('live/offline history differs')
                comparisons+=1; counts['derived_windows_compared']+=1
            rows.append({'scene_id':scene,'compared_windows':comparisons,'status':'PASS'})
            if len(rows)%20==0: print(json.dumps({'event':'history_streams_checked','completed':len(rows),'total':120}),flush=True)
        if len(rows)!=120 or counts['derived_windows_compared']+counts['noncanonical_initial_windows_excluded']!=len(windows):
            raise ValueError('incomplete stream/window population')
        if bindings!={name:sha(ROOT/name) for name in sources}: raise ValueError('stream source changed')
        result={'status':'PASS','branches':len(rows),**counts,'rows':rows,'launch_sha256':sha(output/'launch.json'),
            'scope':'live/offline input equivalence only; noncanonical initial RGB deliberately not substituted; no policy decisions'}
        write(output/'result.json',result); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    except Exception as exc:
        write(output/'failure.json',{'status':'FAILED_STREAM_REPLAY','error':repr(exc),'completed_branches':len(rows),**counts}); raise


if __name__=='__main__': main()
