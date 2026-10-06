"""Read-only turn-latch diagnosis of the two completed V2 failures."""
import hashlib,json,math
from collections import Counter
from pathlib import Path
import numpy as np
from lewm import decision_headroom_json_v42_development as output
B=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001')
ROOT=B/'v2_turn_diagnosis_attempt001'
def main():
    ROOT.mkdir(exist_ok=False);output.install(ROOT);rows=[]
    for i,lo,hi in [(1,144.,480.),(9,90.,390.)]:
        r=B/f'runs/v2_exhausted_view_screen_C1_dev{i:02d}_ep0_attempt001'
        p=json.loads((r/'planning.json').read_text());p=[x for x in p if 'selection'in x and lo<=x['frame']*.1<hi]
        q=json.loads((r/'requests.json').read_text());q=[x for x in q if lo<=x['simulator_ns']/1e9-1.5<hi]
        commands=np.array([x['applied_command'] for x in q]);sign=np.sign(commands[:,2]);nonzero=sign[sign!=0]
        with np.load(r/'native/physics_trace.npz') as f:
            mask=(f['timestamp_s']>=lo+1.5)&(f['timestamp_s']<=hi+1.5);poses=f['base_pose_world'][mask][::50]
        x,y,z,w=poses[:,3:].T;yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        witness=[a for a in p if a['action']=='hold' and a['selection'].get('visual_route_turn_memory',{}).get('active')]
        row=dict(episode=i,window=[lo,hi],selected_plans=len(p),route_action_counts={str(k):v for k,v in Counter((a['route_status'],a['action'])for a in p).items()},
            translation_s=.02*int(np.any(commands[:,:2]!=0,axis=1).sum()),turn_only_s=.02*int(((commands[:,2]!=0)&~np.any(commands[:,:2]!=0,axis=1)).sum()),
            holds_s=.02*int(np.all(commands==0,axis=1).sum()),turn_direction_reversals=int(np.sum(nonzero[1:]!=nonzero[:-1])),
            absolute_yaw_travel_rad=float(np.abs(np.diff(yaw)).sum()),net_yaw_change_rad=float(yaw[-1]-yaw[0]),
            net_xy_displacement_m=float(np.linalg.norm(poses[-1,:2]-poses[0,:2])),
            visual_memory_active_count=sum(a['selection'].get('visual_route_turn_memory',{}).get('active',False) for a in p),
            clearance_turn_active_count=sum(a['selection'].get('clearance_turn',{}).get('active',False)for a in p),
            clearance_turn_latches=sum(a['selection'].get('clearance_turn',{}).get('event')=='CLEAR_ALTERNATIVE_TURN_LATCHED'for a in p),
            latched_holds=len(witness),first_latched_hold=None if not witness else dict(frame=witness[0]['frame'],selection=witness[0]['selection']),
            input_sha256={n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in ['planning.json','requests.json','native/physics_trace.npz']})
        rows.append(row);print({k:v for k,v in row.items()if k not in ['first_latched_hold','input_sha256','route_action_counts']},flush=True)
    with (ROOT/'result.json').open('x')as f:json.dump(dict(no_physics=True,no_models=True,rows=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
if __name__=='__main__':main()
