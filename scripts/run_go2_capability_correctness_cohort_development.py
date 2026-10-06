"""Execute one fixed corrected cohort; preserve every failure and re-project."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_correctness_c1_development as owner
from scripts.project_go2_capability_correctness_budget_development import project


def run(stage):
    protocol=json.loads(owner.PROTOCOL.read_text());base=Path(protocol['output_root'])
    output.install(base);budget=owner.Budget(base,protocol)
    arm='C1' if stage=='screen' else 'C0'
    name=f'v0_task_c1_{arm}_{stage}';root=base/'cohorts'/name;root.mkdir(parents=True,exist_ok=False)
    assignments=[(i,j,f'v0_task_c1_{stage}_{arm}_dev{i:02d}_ep{j}_attempt001')
        for i in range(10) for j in ((0,) if stage=='screen' else (0,1))]
    owner.save(root/'config.json',dict(assignments=assignments,harness_sha256=owner.sha(owner.FREEZE),
        owner_sha256=owner.sha(__file__),protocol_sha256=owner.sha(owner.PROTOCOL),automatic_retry=False))
    completed=[];rows=[];started=time.monotonic()
    for maze,episode,assignment in assignments:
        budget.check(force=True)
        destination=base/'runs'/assignment
        with (root/f'{assignment}.log').open('x') as log:
            result=subprocess.run([sys.executable,'scripts/run_go2_navigation_capability_correctness_c1_development.py',
                '--controller',arm,'--maze',str(maze),'--episode',str(episode),'--assignment',assignment],stdout=log,stderr=subprocess.STDOUT)
        if not (destination/'result.json').exists():raise RuntimeError('Owner failed before preserved closeout: '+assignment)
        if result.returncode:
            failure=json.loads((destination/'result.json').read_text())['error'] or ''
            # Only recognised controller pose/tracking failures are navigation outcomes.
            if not any(key in failure.lower() for key in ('pose','tracking','registration')):
                raise RuntimeError('Technical/resource/fidelity stop: '+assignment+' '+failure)
        with (root/f'{assignment}_reader.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/read_go2_navigation_capability_correctness_c1_development.py',
                '--root',str(destination)],stdout=log,stderr=subprocess.STDOUT,check=True)
        evaluation=json.loads((destination/'episode_evaluation.json').read_text())
        if evaluation.get('status')=='STARTUP_FAILURE':raise RuntimeError('No scientific episode: '+assignment)
        completed.append(destination)
        rows.append(dict(assignment=assignment,episode_id=evaluation['episode_id'],
            round_trip_success=evaluation['round_trip_success'],disallowed_contacts=evaluation['disallowed_contact_samples'],
            hard_violations=evaluation['safety']['hard']['confirmed_violation_samples'],
            hard_unresolved=evaluation['safety']['hard']['unresolved_sampled_samples'],
            failure_and_stall_taxonomy=evaluation['failure_and_stall_taxonomy'],
            wall_s=evaluation['wall_s']))
        owner.save(root/f'{assignment}_result.json',rows[-1])
        if arm=='C0' and len(rows)==1:
            with (root/'C0_sensor_replay.log').open('x') as log:
                subprocess.run([sys.executable,'scripts/check_go2_capability_corrected_C0_replay_development.py'],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        if rows[-1]['disallowed_contacts'] or rows[-1]['hard_violations']:
            break
    successes=sum(r['round_trip_success'] for r in rows)
    passed=len(rows)==len(assignments) and successes>=(9 if arm=='C1' else 19) and all(
        r['disallowed_contacts']==r['hard_violations']==r['hard_unresolved']==0 for r in rows)
    owner.save(root/'result.json',dict(controller=arm,harness='v0_task_c1',harness_sha256=owner.sha(owner.FREEZE),
        complete=len(rows)==len(assignments),episodes=len(rows),successes=successes,passed=passed,
        rows=rows,wall_s=time.monotonic()-started,stop_on_safety_disqualification=len(rows)<len(assignments)))
    prior=[]
    if stage=='gate':
        old=json.loads((base/'cohorts/v0_task_c1_C1_screen/result.json').read_text())
        prior=[base/'runs'/r['assignment'] for r in old['rows']]
    projection=project(prior+completed);owner.save(root/'budget_projection.json',projection)
    print(json.dumps(dict(cohort=name,passed=passed,successes=successes,episodes=len(rows),
        projected_total_wall_hours=projection['projected_total_wall_hours'],storage_fits=projection['storage_fits'])),flush=True)
    if not projection['time_fits'] or not projection['storage_fits']:raise RuntimeError('Post-cohort budget stop')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','gate'],required=True)
    run(p.parse_args().stage)
