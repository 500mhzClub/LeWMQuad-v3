"""Fixed two/four-owner C1 replay checks on its original pilot episode."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from lewm import decision_headroom_json_v42_development as output
from scripts.run_go2_navigation_capability_development import Budget, PROTOCOL, save, sha


def main():
    protocol=json.loads(PROTOCOL.read_text());base=Path(protocol['output_root']);output.install(base)
    source=base/'runs/v0_pilot_C1_dev00_ep0_attempt001'
    original=base/'videos/pipeline_test_attempt001/replay_verification.json'
    assert json.loads(original.read_text())['unused_workload_equivalence_passed']
    pilots={arm:base/f'runs/v0_pilot_{arm}_dev00_ep0_attempt{2 if arm=="C0" else 1:03d}/result.json'
        for arm in ['C0','C1','C2','C3','C4']}
    if not all(p.exists() for p in pilots.values()):
        raise ValueError('Complete the one-per-controller serial pilots before the concurrency checks')
    root=base/'concurrency_C1_v0_attempt001';root.mkdir(exist_ok=False)
    budget=Budget(base,protocol);budget.admit_persist(512*1024**2)
    script='scripts/check_go2_navigation_capability_replay_development.py'
    plan=dict(schema='navigation_capability_concurrency.v1',controller='C1',source=str(source),
        source_config_sha256=sha(source/'config.json'),serial_equivalence_sha256=sha(original),
        assignments={str(n):[f'C1_concurrent_{n}_{i}_attempt001' for i in range(n)] for n in [2,4]},
        levels=[2,4],all_six_checks_fixed_before_execution=True,
        completed_serial_pilots={arm:dict(path=str(p),sha256=sha(p)) for arm,p in pilots.items()},
        devices=dict(physics='CPU',renderer='existing integrated Radeon / renderD129',motion_model='CPU fitted command history'),
        concurrent_C4_training='Allowed only on its unchanged discrete cuda:0 device; no shared GPU compute',
        unchanged_programme_caps=protocol['caps'],admission='Every owner must reproduce all original RGB, decisions, commands and native physics exactly',
        implementation_only=True,source_episode_repeated_as_science=False,
        bindings={p:sha(p) for p in [__file__,script,'scripts/render_go2_navigation_capability_pipeline_development.py',
            'lewm/navigation_capability_unused_workload_development.py']})
    save(root/'plan.json',plan)
    levels=[]
    for n in plan['levels']:
        budget.check(force=True);started=time.monotonic();workers=[]
        for name in plan['assignments'][str(n)]:
            log=(root/f'{name}.log').open('x')
            process=subprocess.Popen([sys.executable,'-u',script,'--source-root',str(source),'--assignment',name],stdout=log,stderr=subprocess.STDOUT)
            workers.append((name,process,log))
        codes=[]
        for name,process,log in workers:
            codes.append(dict(assignment=name,returncode=process.wait()));log.close()
        elapsed=time.monotonic()-started;results=[]
        for row in codes:
            path=base/'equivalence'/row['assignment']/'result.json'
            if path.exists():results.append(json.loads(path.read_text()))
        passed=len(results)==n and all(r['returncode']==0 for r in codes) and all(r['passed'] and r['exact_native_trace_values'] for r in results)
        level=dict(owners=n,passed=passed,wall_s=elapsed,returncodes=codes,
            aggregate_simulated_s=sum(r['simulated_s'] for r in results),results=results)
        save(root/f'level_{n}.json',level);levels.append(level)
        print(output.dumps({k:v for k,v in level.items() if k not in ['results','returncodes']}),flush=True)
        if not passed:break
    save(root/'result.json',dict(passed=len(levels)==2 and all(r['passed'] for r in levels),levels=levels,
        admission_limited_to='C1 CPU motion model with original integrated-GPU renderer',
        other_controller_concurrency_qualified=False,automatic_retry=False))


if __name__=='__main__':main()
