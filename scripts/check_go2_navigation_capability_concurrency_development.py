"""Fixed two/four-owner replay checks, preferably on the learned GPU pathway."""
import json
from pathlib import Path
import subprocess
import sys
import time

from lewm import decision_headroom_json_v42_development as output
from scripts.run_go2_navigation_capability_development import Budget, PROTOCOL, save, sha


def main():
    protocol=json.loads(PROTOCOL.read_text());base=Path(protocol['output_root']);output.install(base)
    pilots={arm:base/f'runs/v0_pilot_{arm}_dev00_ep0_attempt{2 if arm=="C0" else 1:03d}/result.json'
        for arm in ['C0','C1','C2','C3','C4']}
    if not all(p.exists() for p in pilots.values()):
        raise ValueError('Complete the one-per-controller serial pilots before the concurrency checks')
    # Fixed before C3 finishes: use its GPU workload if its recording is
    # complete. A mission budget exhaustion is eligible; success is irrelevant.
    c3=json.loads(pilots['C3'].read_text())
    arm='C3' if c3['error'] is None and c3['policy_steps']>0 else 'C1'
    source=pilots[arm].parent
    root=base/f'concurrency_{arm}_v0_attempt001';root.mkdir(exist_ok=False)
    budget=Budget(base,protocol);budget.admit_persist(512*1024**2)
    script='scripts/check_go2_navigation_capability_replay_development.py'
    plan=dict(schema='navigation_capability_concurrency.v1',controller=arm,source=str(source),
        source_choice='C3 if a complete recording without source exception exists; otherwise the retained C1 trace. Never choose by round-trip success or replace a science trial.',
        source_config_sha256=sha(source/'config.json'),serial_source_result_sha256=sha(pilots[arm]),
        assignments={str(n):[f'{arm}_concurrent_{n}_{i}_attempt001' for i in range(n)] for n in [2,4]},
        levels=[2,4],all_six_checks_fixed_before_execution=True,
        completed_serial_pilots={arm:dict(path=str(p),sha256=sha(p)) for arm,p in pilots.items()},
        devices=dict(physics='CPU',renderer='existing integrated Radeon / renderD129',motion_model='cuda:0 R9700' if arm=='C3' else 'CPU fitted command history'),
        concurrent_C4_training=False,
        timing_caveat='Source baseline includes recording; concurrent checks include exact verification. Use measured concurrent wall costs, not an isolated latency-effect claim.',
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
        admission_limited_to=f'{arm} on these unchanged devices; other controller types remain serial',
        other_controller_concurrency_qualified=False,automatic_retry=False))


if __name__=='__main__':main()
