#!/usr/bin/env python3
"""Full actual composite tensor check after the384-trial independent raw audit."""
import builtins
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.moving_prefix_learning_data_development import AuditedMovingPrefixDataset,OUTPUT as PHYSICAL,LAUNCH_SHA
from scripts.run_go2_successive_choice_maze_development_v1 import digest,verify,write_json

OUTPUT=ROOT/'.generated/go2_moving_prefix_tensor_check_development_v1_attempt_001'
NEW_SOURCES=('lewm/moving_prefix_learning_data_development.py','lewm/tests/test_moving_prefix_learning_data_development.py',
    'scripts/check_go2_moving_prefix_tensors_development_v1.py','docs/go2_moving_prefix_tensor_check_development_v1_2026-09-05.md')


def tensor_hash(value): return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def main():
    if len(sys.argv)!=1 or OUTPUT.exists(): raise ValueError('fixed fresh full composite check required')
    if digest(PHYSICAL/'launch.json')!=LAUNCH_SHA: raise ValueError('exact physical launch required')
    launch=json.loads((PHYSICAL/'launch.json').read_text()); audit=json.loads((PHYSICAL/'raw_artifact_audit.json').read_text())
    if audit['status']!='PASS' or audit['audited_trials']!=384 or audit['study_result_sha256']!=digest(PHYSICAL/'result.json'):
        raise ValueError('full independent physical audit required')
    sources=launch['source_sha256']|audit['audit_source_sha256']|{p:digest(ROOT/p) for p in NEW_SOURCES}
    inputs={str((PHYSICAL/n).relative_to(ROOT)):digest(PHYSICAL/n) for n in ('launch.json','result.json','raw_artifact_audit.json')}
    inputs.update(launch['input_sha256']); verify(sources|inputs|launch['gait_sha256'])
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',{'source_sha256':sources,'input_sha256':inputs,
        'gait_sha256':launch['gait_sha256'],'scope':'all available600 planned development cells; policy-only reads, conditioning equality and target masks; no fitting'})
    rows=[]; accounting=[]; groups={}; accesses=set()
    old_builtin=builtins.open; old_path=Path.open
    def guard_name(name):
        if isinstance(name,(str,Path)):
            p=Path(name)
            if (any(v in ('sealed','sealed_test.json') or v.startswith('sealed_') for v in p.parts)
                    or p.name in ('physics_trace.npz','camera_audit.json','native_contacts.npz','contact_topology.json','contact_events.json')):
                raise ValueError('privileged/protected file requested by policy tensor path')
            accesses.add(str(p))
    def guarded_builtin(name,*args,**kwargs): guard_name(name); return old_builtin(name,*args,**kwargs)
    def guarded_path(path,*args,**kwargs): guard_name(path); return old_path(path,*args,**kwargs)
    try:
        with patch('builtins.open',guarded_builtin),patch.object(Path,'open',guarded_path):
            for role in ('train','validation'):
                data=AuditedMovingPrefixDataset(PHYSICAL,role)
                if len(data)+len(data.unavailable)!=len(data.cells): raise ValueError('planned-cell accounting')
                accounting.append({'role':role,'planned':len(data.cells),'available':len(data),'unavailable':data.unavailable})
                for item in (data[i] for i in range(len(data))):
                    history=item['observation_history']; target=item['targets']; meta=item['metadata']
                    if set(history)!={'rgb','body','control'} or any(not torch.isfinite(v).all() for v in history.values()): raise ValueError('policy modality or finiteness')
                    if {k:tuple(v.shape) for k,v in history.items()}!={'rgb':(4,3,96,128),'body':(4,20,63),'control':(4,15,7)}:
                        raise ValueError('causal history tensor shapes')
                    expected_ticks=40 if meta['prefix_action_index']==0 else 30
                    mask=item['known_action_valid']; blocks=item['known_action_blocks']
                    if mask.shape!=(8,5) or blocks.shape!=(8,5,3) or not torch.equal(mask.flatten(),torch.arange(40)<expected_ticks): raise ValueError('fixed prospective plan mask')
                    if not torch.isfinite(blocks).all() or torch.count_nonzero(blocks[~mask]): raise ValueError('unknown plan content')
                    mv,cv=target['motion_valid'],target['contact_valid']
                    future=target['future_observations']
                    if set(future)!=set(history) or any(v.shape!=(8,*history[k].shape[1:]) or not torch.isfinite(v).all() for k,v in future.items()):
                        raise ValueError('future packet tensor shapes or modalities')
                    if (not torch.equal(target['future_valid'],mv) or (mv&~cv).any() or (cv&~mask.all(-1)).any()
                            or not torch.isfinite(target['motion'][mv]).all() or not torch.isfinite(target['contact'][cv]).all()
                            or not torch.isnan(target['motion'][~mv]).all() or not torch.isnan(target['contact'][~cv]).all()):
                        raise ValueError('actual target censoring')
                    if torch.any(target['contact'][mv]!=0) or not torch.all((target['contact'][cv]==0)|(target['contact'][cv]==1)):
                        raise ValueError('contact semantics')
                    if any(torch.count_nonzero(v[~mv]) for v in target['future_observations'].values()): raise ValueError('censored future content')
                    hashes={k:tensor_hash(v) for k,v in history.items()}; key=meta['conditioning_group']
                    if key in groups and groups[key]['hashes']!=hashes: raise ValueError('counterfactual siblings have different model conditioning')
                    if key not in groups: groups[key]={'role':role,'hashes':hashes,'actions':[]}
                    if groups[key]['role']!=role or meta['action_index'] in groups[key]['actions']: raise ValueError('group action duplication or role leakage')
                    groups[key]['actions'].append(meta['action_index'])
                    rows.append({'metadata':meta,'history_sha256':hashes,'motion_valid':int(mv.sum()),'contact_valid':int(cv.sum()),
                        'contact_positive':int(target['contact'][cv].sum())})
                print(json.dumps({'event':'composite_role_checked','role':role,'available':len(data),'unavailable':len(data.unavailable)}),flush=True)
        if len(groups)!=120: raise ValueError('24 layouts times five conditioning groups required')
        verify(sources|inputs|launch['gait_sha256'])
        result={'status':'PASS','checked_cells':len(rows),'planned_cells':600,'conditioning_groups':len(groups),'accounting':accounting,
            'groups':groups,'cells':rows,'policy_path_forbidden_reads':0,'observed_file_accesses':sorted(accesses),
            'launch_sha256':digest(OUTPUT/'launch.json'),'scope':'actual tensor semantics only; no model fitting or navigation claim'}
        write_json(OUTPUT/'result.json',result); print(json.dumps({k:result[k] for k in ('status','checked_cells','conditioning_groups')}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',{'status':'FAIL','error':str(error),'checked_cells':len(rows),'cells':rows,
            'launch_sha256':digest(OUTPUT/'launch.json')}); raise


if __name__=='__main__': main()
