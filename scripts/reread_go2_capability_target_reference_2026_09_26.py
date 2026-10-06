"""No-physics reread; never overwrite the pre-fix pilot reports."""
import json
from pathlib import Path
import numpy as np
from lewm import decision_headroom_json_v42_development as output
from lewm.navigation_capability_target_reference_development import fixed_world_arrivals
from scripts.correct_go2_navigation_capability_leg_accounting_development import legs
from scripts.run_go2_navigation_capability_development import PROTOCOL, save, sha


def main():
    base=Path(json.loads(PROTOCOL.read_text())['output_root']);output.install(base)
    rows=[]
    for arm in ['C0','C1','C2','C3','C4']:
        root=base/f'runs/v0_pilot_{arm}_dev00_ep0_attempt{2 if arm=="C0" else 1:03d}'
        prior=root/'episode_evaluation_leg_accounting_v2.json'
        old=json.loads(prior.read_text());episode=json.loads((root/'episode.json').read_text())
        mission=json.loads((root/'mission.json').read_text());requests=json.loads((root/'requests.json').read_text())
        with np.load(root/'native/physics_trace.npz',allow_pickle=False) as a:
            trace={k:a[k].copy() for k in ('base_pose_world','timestamp_s')}
        meta=root/'native/in_memory_camera_observations.json'
        if meta.exists():frames=json.loads(meta.read_text())['frames']
        else:
            stamps=np.rint(trace['timestamp_s']*1e9).astype(np.int64)
            frames=[r|dict(physical_sample_index=int(np.searchsorted(stamps,r['measured_ns'])))
                for r in json.loads((root/'acquisitions.json').read_text())]
            assert all(stamps[r['physical_sample_index']]==r['measured_ns'] for r in frames)
        arrivals=fixed_world_arrivals(episode,mission,frames,trace,requests)
        passed={r['phase']:r['passed'] for r in arrivals}
        lookup={r['frame']:r['physical_sample_index'] for r in frames}
        boundaries={r['phase']:lookup[r['frame']] for r in arrivals}
        beacon,home=passed.get('OUTBOUND',False),passed.get('RETURN',False)
        outbound,return_leg=legs(trace,frames[0]['physical_sample_index'],boundaries.get('OUTBOUND'),
            boundaries.get('RETURN'),beacon,home,[episode['shortest_outbound_m'],episode['shortest_return_m']])
        record=old|dict(schema='navigation_capability_episode_evaluation.v3',arrivals=arrivals,
            beacon_success=beacon,home_success=home,round_trip_success=beacon and home,
            outbound=outbound,return_leg=return_leg,label='PRE-FIX development pilot; corrected fixed-world reader; not capability evidence',
            target_reference_erratum=dict(new_physics=False,original_preserved=True,original_sha256=sha(prior),
                code_sha256=sha('lewm/navigation_capability_target_reference_development.py'),
                safety_reused_unchanged=True,geometry='registered fixed world beacon and home',
                obsolete_initial_body_distance_gate_removed=True))
        save(root/'episode_evaluation_reference_v3.json',record)
        rows.append(dict(controller=arm,beacon_success=beacon,home_success=home,round_trip_success=beacon and home,
            verdict_changed=(beacon,home)!=(old['beacon_success'],old['home_success']),
            maximum_arrival_distance_m={r['phase']:r['native_maximum_distance_m'] for r in arrivals},
            report=str(root/'episode_evaluation_reference_v3.json')))
    save(base/'pilot_reference_reread_2026-09-26.json',dict(pre_fix_pilots=True,new_physics=False,rows=rows))
    print(output.dumps(rows))


if __name__=='__main__':main()
