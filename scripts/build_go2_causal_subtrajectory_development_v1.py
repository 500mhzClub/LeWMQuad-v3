#!/usr/bin/env python3
"""Derive fixed temporal windows from exact audited development artifacts; no fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lewm.causal_subtrajectory_development import branch_windows
from lewm.causal_subtrajectory_learning_development import CORPUS_ROOT,DERIVATION_ROOT
from lewm.counterfactual_learning_data_development import AuditedCounterfactualDataset

SOURCE_PATHS=('lewm/causal_subtrajectory_development.py','lewm/causal_subtrajectory_learning_development.py',
    'scripts/build_go2_causal_subtrajectory_development_v1.py',
    'docs/go2_causal_subtrajectory_development_v1_2026-09-05.md',
    'lewm/tests/test_causal_subtrajectory_development.py')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def checked_leaf(directory,name,expected):
    if Path(name).name!=name or name in ('sealed_test.json','sealed') or name.startswith('sealed_'):
        raise ValueError('invalid explicit development artifact')
    path=directory/name
    if path.resolve().parent!=directory.resolve() or digest(path)!=expected:
        raise ValueError('artifact binding/path changed')
    return path


def source_bindings():
    launch=json.loads((CORPUS_ROOT/'launch.json').read_text())
    bindings={}
    # Existing source identities are carried forward, not clean exported or edited.
    for name,value in launch['source_sha256'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or any(s in ('sealed','sealed_test.json') or s.startswith('sealed_') for s in path.parts):
            raise ValueError('invalid source binding')
        if digest(ROOT/path)!=value: raise ValueError('corpus source changed')
        bindings[name]=value
    for name in SOURCE_PATHS: bindings[name]=digest(ROOT/name)
    for name in ('lewm/counterfactual_learning_data_development.py','lewm/rgb_body_tensor_interface_development.py'):
        bindings[name]=digest(ROOT/name)
    return bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    if output!=DERIVATION_ROOT or output.resolve()!=output or output.exists(): raise ValueError('fresh exact derivation root required')
    data=AuditedCounterfactualDataset(CORPUS_ROOT,'train')
    report=json.loads((CORPUS_ROOT/'result.json').read_text())
    bindings=source_bindings()
    launch={'schema':'causal_subtrajectory_development.v1','corpus_result_sha256':digest(CORPUS_ROOT/'result.json'),
        'corpus_audit_sha256':digest(CORPUS_ROOT/'raw_artifact_audit.json'),'source_sha256':bindings,
        'scope':'target derivation only; fixed development corpus; no training or new physics'}
    output.mkdir(); write_json(output/'launch.json',launch)
    windows=[]; branch_bindings={}
    try:
        for member in report['members']:
            row=member['result']
            if not row['branchable']: continue
            scene=row['scene_id']; directory=data.paths[scene]
            path=checked_leaf(directory,'physics_trace.npz',row['artifact_sha256']['physics_trace.npz'])
            with np.load(path,allow_pickle=False) as archive:
                raw={k:archive[k] for k in ('timestamp_s','base_pose_world','physics_contact','phase','applied_command')}
            frames=json.loads((directory/'policy_observations.json').read_text())['frames']
            canonical=member['prefix_match']['canonical_model_context_scene_id']
            if data.members[canonical]['result']['layout_id']!=row['layout_id']:
                raise ValueError('canonical context crossed layout')
            canonical_frames=json.loads((data.paths[canonical]/'policy_observations.json').read_text())['frames']
            branch=branch_windows(raw,row['prefix_terminal_sample_index'],frames,canonical_frames)
            for window in branch:
                windows.append({**window,'window_id':f'{scene}-t{window["offset_ns"]}',
                    'scene_id':scene,'layout_id':row['layout_id'],'data_role':row['data_role'],
                    'action_index':row['action_index'],'context_scene_id':canonical if window['offset_ns']==0 else scene})
            branch_bindings[scene]={'member_result_sha256':member['result_sha256'],
                'physics_sha256':digest(path),'windows':len(branch)}
        write_json(output/'windows.json',windows)
        if source_bindings()!=bindings: raise ValueError('source changed during derivation')
        counts={}
        for role in ('train','validation'):
            rows=[w for w in windows if w['data_role']==role]
            counts[role]={'layouts':len({w['layout_id'] for w in rows}),'windows':len(rows),
                'by_offset_ns':dict(Counter(str(w['offset_ns']) for w in rows)),
                'in_plan_horizons':sum(t['in_plan'] for w in rows for t in w['targets']),
                'motion_valid':sum(t['motion_valid'] for w in rows for t in w['targets']),
                'contact_valid':sum(t['contact_valid'] for w in rows for t in w['targets']),
                'contact_positive':sum(t['contact_by_horizon'] is True for w in rows for t in w['targets'])}
        result={'status':'COMPLETE','window_count':len(windows),'branch_count':len(branch_bindings),
            'counts':counts,'branch_bindings':branch_bindings,'windows_sha256':digest(output/'windows.json'),
            'launch_sha256':digest(output/'launch.json'),'corpus_result_sha256':launch['corpus_result_sha256']}
        write_json(output/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='branch_bindings'},indent=2))
    except Exception as exc:
        write_json(output/'failure.json',{'status':'FAILED_DERIVATION','error':repr(exc),'constructed_windows':len(windows)})
        raise


if __name__=='__main__': main()
