"""Prediction-slot additions to the frozen v0 owner; controller code unchanged."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from lewm.eligible_floor_registration_development import bind
from lewm.navigation_capability_direct_slot_development import load as load_direct
from lewm.navigation_capability_unused_workload_development import UnusedNeuralWorkload
from lewm_genesis.lewm_contract import SafetyLimits
from scripts import run_go2_navigation_capability_development as owner

FREEZE=owner.REPO/'docs/go2_navigation_capability_harness_v0_adapter_r3_2026-09-25.json'


def run(arm,maze,episode,assignment,omit_unused=False):
    protocol=json.loads(owner.PROTOCOL.read_text());base=Path(protocol['output_root'])
    evidence=None
    if omit_unused:
        if arm!='C1':raise ValueError('only C1 has completed unused-workload equivalence')
        path=base/'videos/pipeline_test_attempt001/replay_verification.json'
        evidence=json.loads(path.read_text())
        if not evidence['unused_workload_equivalence_passed'] or not evidence['exact_native_trace_values']:
            raise ValueError('complete exact source replay required')
        evidence=dict(path=str(path),sha256=owner.sha(path))
    direct_identity=None
    if arm=='C4':
        path=base/'c4_fit_attempt002/direct_final.pt'
        result=json.loads((path.parent/'result.json').read_text())
        if result['status']!='COMPLETE' or owner.sha(path)!=result['checkpoint_sha256']:
            raise ValueError('fixed final C4 checkpoint required')
        direct_identity=dict(path=str(path),sha256=result['checkpoint_sha256'],
            plan_sha256=owner.sha(path.parent/'plan.json'),training_render_provenance='unverified')

    def load_model(*args,**kwargs):
        if arm=='C4':return load_direct(protocol)
        if omit_unused:
            limits=SafetyLimits.from_manifest(yaml.safe_load((owner.REPO/'config/go2_platform_manifest.yaml').read_text()))
            return UnusedNeuralWorkload(arm,limits)
        return owner.source.load_dense_navigation_model(*args,**kwargs)

    def save(path,value):
        if path.name=='config.json':
            value=dict(value,prediction_slot_addition=direct_identity,
                unused_workload_omitted=omit_unused,equivalence_evidence=evidence,
                software=dict(torch=str(torch.__version__),hip=str(torch.version.hip)),
                devices=[dict(index=i,name=torch.cuda.get_device_name(i),
                    capacity_bytes=torch.cuda.get_device_properties(i).total_memory)
                    for i in range(torch.cuda.device_count())])
        owner.save(path,value)

    source=SimpleNamespace(**(vars(owner.source)|dict(load_dense_navigation_model=load_model)))
    bind(owner.run,source=source,FREEZE=FREEZE,save=save)(arm,maze,episode,assignment)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller',choices=['C0','C1','C2','C3','C4'],required=True)
    p.add_argument('--maze',type=int,choices=range(10),default=0)
    p.add_argument('--episode',type=int,choices=(0,1),default=0)
    p.add_argument('--assignment',required=True);p.add_argument('--omit-unused',action='store_true')
    a=p.parse_args()
    if '/' in a.assignment or a.assignment.startswith('.'):raise ValueError('fresh assignment name required')
    run(a.controller,a.maze,a.episode,a.assignment,a.omit_unused)
