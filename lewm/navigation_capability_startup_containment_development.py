"""Exact trajectory containment of all five non-startup-failed old screen runs."""
import json
from pathlib import Path
import numpy as np

UNAFFECTED_STARTUP_IDS=('00/0','03/0','04/0','05/0','07/0')


def compare(base,new_run):
    episode=json.loads((new_run/'episode.json').read_text())['episode_id']
    if episode not in UNAFFECTED_STARTUP_IDS:
        return dict(episode=episode,required=False,reason='Original pre-planning startup failure')
    maze=int(episode.split('/')[0]);old=base/f'runs/v0_task_c1_screen_C1_dev{maze:02d}_ep0_attempt001'
    differences=[]
    with np.load(old/'native/physics_trace.npz',allow_pickle=False) as a,np.load(new_run/'native/physics_trace.npz',allow_pickle=False) as b:
        for key in sorted(set(a.files)|set(b.files)):
            if key not in a or key not in b or not np.array_equal(a[key],b[key]):differences.append('native/'+key)
    a=json.loads((old/'requests.json').read_text());b=json.loads((new_run/'requests.json').read_text())
    keys=('requested_command','applied_command','reason','simulator_ns')
    if [{k:r.get(k) for k in keys} for r in a]!=[{k:r.get(k) for k in keys} for r in b]:differences.append('requests')
    a=json.loads((old/'native/in_memory_camera_observations.json').read_text())['frames']
    b=json.loads((new_run/'native/in_memory_camera_observations.json').read_text())['frames']
    keys=('consumed_packet_sha256','pixel_sha256','arrays','measured_ns')
    if [{k:r.get(k) for k in keys} for r in a]!=[{k:r.get(k) for k in keys} for r in b]:differences.append('consumed_sensor_hashes')
    return dict(episode=episode,required=True,identical=not differences,differences=differences,
        prior_run=str(old),classification_if_divergent='Count combined correction as a harness version and diagnose cause',
        conservative_population='All original runs that reached planning, including map-extent-potentially-affected runs')
