"""One serial, no-retry command-replay pass over the five authorised originals."""
import json
import subprocess
import sys
from pathlib import Path
from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_correctness_c1_development as owner
from scripts.replay_go2_capability_bound_exposure_development import EPISODES


def main():
    protocol=json.loads(owner.PROTOCOL.read_text());base=Path(protocol['output_root']);output.install(base)
    root=base/'grid_c3_bound_exposure';root.mkdir(exist_ok=False)
    budget=owner.Budget(base,protocol);budget.admit_persist(320*1024**2)
    durations=[]
    for i in EPISODES:
        source=base/f'runs/v0_task_c1_screen_C1_dev{i:02d}_ep0_attempt001'
        durations.append(1.5+.02*len(json.loads((source/'requests.json').read_text())))
    owner.save(root/'config.json',dict(episodes=list(EPISODES),maximum_total_simulated_s=sum(durations),
        wall_cap_hours=160,additional_projected_wall_hours=2.,retained_output_cap_bytes=320*1024**2,
        no_model_fitting=True,no_new_navigation=True,no_retries=True,owner_sha256=owner.sha(__file__),
        replay_owner_sha256=owner.sha('scripts/replay_go2_capability_bound_exposure_development.py'),
        purpose='Actual first legacy-bound exposure for refined containment; stop each replay at first exposure or original endpoint'))
    rows=[]
    for i in EPISODES:
        budget.check(force=True)
        with (root/f'dev{i:02d}.log').open('x') as log:
            result=subprocess.run([sys.executable,'scripts/replay_go2_capability_bound_exposure_development.py','--maze',str(i)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            owner.save(root/'stop.json',dict(maze=i,returncode=result.returncode,completed=rows,automatic_retry=False));raise SystemExit(result.returncode)
        rows.append(json.loads((root/f'dev{i:02d}/result.json').read_text()))
    owner.save(root/'result.json',dict(status='PASS',rows=rows,frames_retained=False,new_navigation_trials=0))

if __name__=='__main__':main()
