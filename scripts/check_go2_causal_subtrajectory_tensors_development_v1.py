#!/usr/bin/env python3
"""Exercise every audited window through the policy-only tensor loader; no model fit."""
from collections import Counter
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lewm.causal_subtrajectory_learning_development import AuditedSubtrajectoryDataset,DERIVATION_ROOT
from scripts.build_go2_causal_subtrajectory_development_v1 import digest,write_json


def check(value,message):
    if not value: raise ValueError(message)


def main():
    output=DERIVATION_ROOT/'tensor_interface_check.json'
    check(not output.exists(),'tensor check already exists')
    count=Counter()
    for role in ('train','validation'):
        data=AuditedSubtrajectoryDataset(DERIVATION_ROOT,role)
        for index in range(len(data)):
            row=data[index]; targets=row['targets']; history=row['observation_history']
            check(set(row)=={'observation_history','known_action_blocks','known_action_valid','targets','metadata'},'top-level tensor interface')
            check(set(history)=={'rgb','body','control'},'observation contains undeclared modality')
            for name,shape in {'rgb':(4,3,96,128),'body':(4,20,63),'control':(4,15,7)}.items():
                check(history[name].shape==shape and torch.isfinite(history[name]).all(),'finite history shape')
                future=targets['future_observations'][name]
                check(future.shape==(8,*shape[1:]) and torch.isfinite(future).all(),'finite target image/history shape')
                check(torch.count_nonzero(future[~targets['future_valid']])==0,'unobserved future padding')
            valid=row['known_action_valid']; plan=row['known_action_blocks']
            check(valid.shape==(8,5) and valid.dtype==torch.bool and plan.shape==(8,5,3),'plan mask shape')
            check(torch.isfinite(plan).all() and torch.count_nonzero(plan[~valid])==0,'plan padding')
            for name in ('motion','contact'):
                mask=targets[f'{name}_valid']
                check(torch.isfinite(targets[name][mask]).all() and torch.isnan(targets[name][~mask]).all(),'censored targets')
                check(not (mask&~valid.all(-1)).any(),'target beyond known plan')
            check(torch.equal(targets['future_valid'],targets['motion_valid']),'future/motion identity')
            check(row['metadata']['data_role']==role,'loader data role')
            count[role]+=1
            if index%100==99: print(json.dumps({'role':role,'checked':index+1,'total':len(data)}),flush=True)
    result={'status':'PASS','windows':dict(count),'study_result_sha256':digest(DERIVATION_ROOT/'result.json'),
        'raw_audit_sha256':digest(DERIVATION_ROOT/'raw_artifact_audit.json'),
        'checker_source_sha256':digest(Path(__file__)),'scope':'actual policy-only loader and masking; no fitting or model outcome'}
    write_json(output,result); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
