"""Finish the fixed pilot queue; stop before any harness/cohort execution."""
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from lewm import decision_headroom_json_v42_development as output
from scripts.run_go2_navigation_capability_development import Budget, PROTOCOL, save, sha


def main():
    protocol=json.loads(PROTOCOL.read_text());base=Path(protocol['output_root']);output.install(base)
    root=base/'pilot_completion_owner_v1';root.mkdir(exist_ok=False)
    budget=Budget(base,protocol)
    c3=base/'runs/v0_pilot_C3_dev00_ep0_attempt001'
    c4=base/'runs/v0_pilot_C4_dev00_ep0_attempt001'
    identity=json.loads((c3/'process.json').read_text())
    jobs=[('C3_native_outcomes',['scripts/read_go2_navigation_capability_episode_development.py','--root',str(c3)]),
        ('C4_serial_pilot',['scripts/run_go2_navigation_capability_slots_development.py','--controller','C4','--assignment',c4.name]),
        ('C4_native_outcomes',['scripts/read_go2_navigation_capability_episode_development.py','--root',str(c4)]),
        ('concurrency',['scripts/check_go2_navigation_capability_concurrency_development.py'])]
    save(root/'plan.json',dict(wait_for_existing_C3=identity,C3_not_restarted=True,jobs=jobs,
        bindings={p:sha(p) for p in [__file__,*[a[0] for _,a in jobs]]},
        fixed_C4_checkpoint_sha256=json.loads((base/'c4_fit_attempt002/result.json').read_text())['checkpoint_sha256'],
        caps=protocol['caps'],after_queue='Stop for measured budget assessment; no cohort or harness iteration in this owner',
        automatic_retries=False))
    def stopped():
        return (base/'STOP_BEFORE_NEXT_ASSIGNMENT').exists()
    while True:
        budget.check()
        if stopped():save(root/'stop.json',dict(reason='Explicit stop marker'));return
        try:
            p=psutil.Process(identity['pid'])
            alive=p.is_running() and abs(p.create_time()-identity['created'])<.1 and p.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:alive=False
        if not alive:break
        time.sleep(5)
    if not (c3/'result.json').exists():
        save(root/'stop.json',dict(reason='C3 process ended without a result; inspect preserved logs'));return
    result=json.loads((c3/'result.json').read_text())
    if result['error'] is not None:
        save(root/'stop.json',dict(reason='Classify C3 source exception before the next assignment',source_error=result['error'],
            controller_failure_is_not_programme_stop=True));return
    for name,args in jobs:
        budget.check(force=True)
        if stopped():save(root/'stop.json',dict(reason='Explicit stop marker',before_job=name));return
        if name=='C3_native_outcomes' and (c3/'episode_evaluation.json').exists():continue
        with (root/f'{name}.log').open('x') as log:
            started=time.monotonic()
            code=subprocess.run([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT).returncode
        save(root/f'{name}_completion.json',dict(returncode=code,wall_s=time.monotonic()-started))
        print(output.dumps(dict(job=name,returncode=code)),flush=True)
        if code:
            save(root/'stop.json',dict(reason='Inspect preserved job failure; no automatic retry',job=name));return
    save(root/'complete.json',dict(pilot_queue_finished=True,cohorts_launched=False,budget_assessment_pending=True))


if __name__=='__main__':main()
