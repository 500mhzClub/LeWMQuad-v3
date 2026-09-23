"""Approval-bound V4 owner. Writing/freezing this file does not authorize execution."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import time
import traceback
import psutil
from scripts.run_go2_decision_headroom_pilot_development import PilotBudget,save

CONFIG=Path('docs/go2_decision_headroom_protocol_v4_2026-09-23.json')

class AuditBudget(PilotBudget):
    def start_source(self,case):
        if type(case)is not int or case not in range(24) or case in self.sources or self.active_case is not None:raise ValueError('fixed unique source assignment required')
        self.check('source_admission',force=True);self.sources.add(case);self.source_ns[case]=0;self.active_case=case;self.event('source_started',case=case)
    def reserve_branch(self,identity):
        if identity in self.branches or len(self.branches)>=6096:raise ValueError('fixed branch cap or duplicate')
        self.check('branch_admission',force=True);self.branches.add(identity);self.event('branch_reserved',identity=identity,physics_ns=800_000_000)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--approval',type=Path,required=True);args=parser.parse_args()
    config=json.loads(CONFIG.read_text());approval=json.loads(args.approval.read_text());digest=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    if approval.get('protocol_sha256')!=digest or approval.get('phase2_explicitly_approved') is not True or not approval.get('user_approval_text'):
        raise RuntimeError('Explicit user approval of this frozen V4 identity is required')
    for p,binding in config['frozen_bindings'].items():
        with Path(p).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=binding['sha256']:raise ValueError('frozen source/protocol identity differs: '+p)
    for binding in config['execution_input_bindings'].values():
        p=Path(binding['path'])
        with p.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=binding['sha256'] or p.stat().st_size!=binding['bytes']:raise ValueError('frozen model/input differs: '+str(p))
    caps=config['execution_caps'];root=Path(caps['output_root'])
    if shutil.disk_usage(root.parent).free < caps['storage_caps']['peak_additional_bytes']+caps['storage_caps']['recovery_filesystem_reserve_bytes']:
        raise RuntimeError('Insufficient free capacity for frozen peak footprint plus reserve; no retirement authorized')
    root.mkdir(exist_ok=False)
    save(root/'explicit_approval.json',approval)
    os.sched_setaffinity(0,caps['compute_caps']['cpu_affinity'])
    admission=dict(owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),caps_sha256=digest,cache_paths=config['cache_paths'],restoration_tolerances=config['restoration_tolerances'])
    save(root/'pilot_execution_admission.json',admission);budget=AuditBudget(root,caps,admission);error=None;outcomes=[]
    try:
        budget.check('initial_admission',force=True)
        from scripts.run_go2_headroom_v4_source_development import run,require_stage_a_closed
        require_stage_a_closed(root)
        for case in range(24):
            try:run(case);outcomes.append(dict(case=case,status='complete'))
            except Exception as exc:
                outcomes.append(dict(case=case,status='unresolved',reason=repr(exc),traceback=traceback.format_exc()))
                if budget.stopped:raise
                budget.active_case=None
            save(root/f'cell_{case:02d}_closeout.json',outcomes[-1])
        save(root/'collection_result.json',dict(status='FIXED_ASSIGNMENTS_CLOSED',cells=outcomes,retries=False,source_navigation_outcomes_descriptive_only=True))
        from scripts.read_go2_headroom_v4_development import report
        report(root,budget=budget)
        budget.check('analysis_complete',force=True)
    except BaseException as exc:
        error=exc;save(root/'failure.json',dict(reason=repr(exc),traceback=traceback.format_exc(),cells=outcomes,no_extension=True))
    finally:
        budget.finish(error)
        # PilotBudget is reused as a meter; correct its legacy authority label.
        p=root/'resource_result.json';r=json.loads(p.read_text());r['phase2_authorized']=True;r['approval_sha256']=hashlib.sha256(args.approval.read_bytes()).hexdigest();p.write_text(json.dumps(r,indent=2)+'\n')
    if error:raise error

if __name__=='__main__':main()
