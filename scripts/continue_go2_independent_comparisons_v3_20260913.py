"""Start the remaining unattempted comparisons after the shared-disk guard failure."""
import json
from pathlib import Path
import subprocess
import sys
import psutil
from scripts.continue_go2_independent_comparisons_20260913 import BASE,CASES,STANDARD,CONTACT


def main():
    cases=CASES[1:]
    for layout,mode,model in cases:
        root=BASE/f'go2_stop_conditioned_independent_{layout:02d}_{mode}_{model}_v1_attempt_001'
        if root.exists() or root.is_symlink():raise ValueError(f'preserve existing attempt: {root}')
    reactive=BASE/'go2_stop_conditioned_independent_00_six_action_reactive_no_model_v1_attempt_001'
    if reactive.exists() or reactive.is_symlink():raise ValueError('preserve reactive attempt')
    owner=psutil.Process()
    receipt=dict(pid=owner.pid,created=owner.create_time(),cases=cases,
        final_six_action_reactive=True,serial_collection_and_audit=True,
        interrupted_direct_audit_preserved=True,automatic_retry=False)
    with Path('docs/go2_independent_comparisons_continuation_v3_2026-09-13.json').open('x') as f:
        json.dump(receipt,f,indent=2)
    print(json.dumps(receipt),flush=True)
    for layout,mode,model in cases:
        script=CONTACT if mode=='commitment_contact' else STANDARD
        args=[sys.executable,'-B',script,'--layout',str(layout),'--model',model]
        if mode!='commitment_contact':args+=['--mode',mode]
        print('STARTING',layout,mode,model,flush=True)
        subprocess.run(args,check=True)
    subprocess.run([sys.executable,'-B','scripts/run_go2_six_action_reactive_independent_case_v1.py'],check=True)


if __name__=='__main__':main()
