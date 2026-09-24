"""Check saved actions against stopping projections during a long approach."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from lewm.planned_stopping_projection_development import stopping_projection_checks
from lewm.two_cm_floor_extent_development import configure
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.reconstruct_go2_frontier_stall_development import RecordedPoseMap, save


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root-name', required=True)
    root = path(parser.parse_args().root_name)
    if (root/'depth_retention.json').exists(): raise ValueError('full retained depth required')
    event = max(read(root, 'frontier_visits.json')['events'],
        key=lambda e:e['completed_ns']-e['started_ns'])
    output = root/'planned_stopping_projection_replay_v1'; output.mkdir()
    plans = defaultdict(list)
    for p in read(root, 'planning.json'):
        if 'selection' in p and event['started_ns'] <= p['measured_ns'] < event['completed_ns']:
            plans[p['map_frame']].append(p)
    requests = read(root, 'requests.json')
    vetoed = {r.get('command_observation_ns') for r in requests if r['reason']=='CURRENT_STOPPING_MARGIN_VETO'}
    poses = {r['frame']:r['registered_pose'] for r in read(root, 'poses.json')}
    maps = sorted((r for r in read(root, 'stage_events.json')
        if r['stage']=='mapping' and r['frame'] <= max(plans)), key=lambda r:r['completed_ns'])
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    reader = NoisyPublicReplay(root/'native'); mapper = RecordedPoseMap(); rows = []; began=time.monotonic()
    for update in maps:
        frame=update['frame']; p,d,_,_,aux,now=reader.packet(frame)
        snapshot=mapper.update(p,d,poses[frame],auxiliary_depth=aux,measured_ns=now)
        B=np.asarray(snapshot.map_from_initial)
        for plan in plans[frame]:
            scope=plan['selection']['routing_memory_scope']
            assert scope['routing_floor_cells']==len(snapshot.floor)
            assert scope['routing_fine_obstacle_cells']==len(snapshot.fine_occupied)
            pose=poses[plan['frame']]; q=B@np.asarray(pose['position_initial_body_m'])
            Q=B@np.asarray(pose['rotation_initial_body_from_current_body'])
            correction=plan['motion_correction']
            checks=stopping_projection_checks(correction['applied_prediction_after_yaw_ablation'],
                snapshot.fine_occupied,q,Q,pulse=bool(correction['terminal_translation_pulse']))
            selected=next(c for c in checks if c['action']==plan['action'])
            rows.append(dict(frame=plan['frame'],measured_ns=plan['measured_ns'],
                action=plan['action'],on_time=plan['on_time'],checks=checks,
                selected_translation=selected['translating'],
                selected_projection_blocked=not selected['projection_clear'],
                actual_stopping_veto=plan['measured_ns'] in vetoed))
    report=dict(plans=len(rows),selected_translations=sum(r['selected_translation'] for r in rows),
        selected_translations_blocked=sum(r['selected_projection_blocked'] for r in rows),
        actually_vetoed_plans=sum(r['actual_stopping_veto'] for r in rows),
        actual_vetoes_anticipated=sum(r['actual_stopping_veto'] and r['selected_projection_blocked'] for r in rows),
        nonvetoed_translations_blocked=sum(r['selected_translation'] and not r['actual_stopping_veto'] and r['selected_projection_blocked'] for r in rows),
        original_registered_poses_and_delivered_pixels_used=True,native_state_used=False,
        alternative_navigation_outcome_established=False,dispatch_guards_changed=False,
        elapsed_s=time.monotonic()-began,source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest()
            for s in (__file__,'lewm/planned_stopping_projection_development.py')})
    save(output,'rows.json',rows);save(output,'result.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__': main()
