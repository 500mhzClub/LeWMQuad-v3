"""Development-only failure diagnosis from completed logs; no physics/models."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from lewm import decision_headroom_json_v42_development as output
from lewm.decision_headroom_reference_development import ReferenceGeometry
from scripts.analyse_go2_capability_grid_c3_failures_readonly_development import wall_ray

BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001')
ROOT=BASE/'paired_floor_leg_diagnosis_attempt001'
def read(p): return json.loads(p.read_text())
def cells(xy):
    result=[]
    for p in np.rint(xy/1.3).astype(int).tolist():
        if not result or p!=result[-1]: result.append(p)
    return result

def erase(seq):
    result=[]
    for p in seq:
        if p in result: result=result[:result.index(p)+1]
        else: result.append(p)
    return result

def analyse(i):
    root=BASE/f'runs/v1_paired_floor_screen_C1_dev{i:02d}_ep0_attempt001'
    names=['episode.json','episode_evaluation.json','planning.json','requests.json','specification.json','native/physics_trace.npz']
    ep,ev,plans,requests,spec=[read(root/n) for n in names[:-1]]
    with np.load(root/names[-1],allow_pickle=False) as f:
        t=f['timestamp_s']-1.5; native=f['base_pose_world'].copy()
    end=float(t[-1]); beacon=ev['outbound']['elapsed_s'] if ev['beacon_success'] else None
    # Request rows are exact 20-ms command intervals, starting at simulator_ns.
    qt=np.array([r['simulator_ns']/1e9-1.5 for r in requests]); q=np.array([r['applied_command'] for r in requests])
    qend=np.minimum(qt+.02,end)
    assert np.max(np.abs(np.diff(qt)-.02))<1e-8
    moving=np.any(q!=0,axis=1); translating=np.any(q[:,:2]!=0,axis=1)
    # Recovery labels are planning-time evidence held until the next planning result.
    # They are not unretained framewise recovery-state timestamps.
    pt=np.array([r['completed_ns']/1e9-1.5 if 'completed_ns' in r else r['frame']*.1 for r in plans])
    ix=np.searchsorted(pt,qt,side='right')-1
    recovery=np.array([j>=0 and plans[j].get('route_status')=='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW' for j in ix])
    walls=[dict(center=w['centre_xyz'][:2],size=w['size_xyz'][:2],yaw=w['yaw_rad']) for w in spec['geometry']['wall_boxes']]
    def geometry(target): return ReferenceGeometry(walls,[[-2.1,-2.1],[3.4,3.4]],target,radius_m=.46,clearance_m=.005,resolution_m=.02)
    geos={'OUTBOUND':geometry(ep['beacon_xy_world']),'RETURN':geometry(ep['home_se2_world'][:2])}
    def pose(s):return native[min(np.searchsorted(t,s),len(t)-1)]
    def window(lo,hi,leg):
        dt=np.maximum(0,np.minimum(qend,hi)-np.maximum(qt,lo))
        pp=[r for r in plans if lo<=r['frame']*.1<hi]; ss=[r for r in pp if 'selection' in r]
        points=native[(t>=lo)&(t<=hi),:2]
        hh=[r for r in ss if r['action']=='hold']
        routes=Counter(r.get('route_status',r.get('reason')) for r in pp)
        scope=[r['selection']['routing_memory_scope'] for r in ss if 'routing_memory_scope' in r['selection']]
        aligned=[r for r in hh if r.get('route_status')=='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW' and abs(r['selection'].get('scan_heading_error_rad',99))<=.1]
        return dict(start_s=lo,end_s=hi,duration_s=hi-lo,moving_s=float(dt[moving].sum()),holding_s=float(dt[~moving].sum()),
            translating_s=float(dt[translating].sum()),turn_only_s=float(dt[moving&~translating].sum()),
            visual_recovery_s=float(dt[recovery].sum()),visual_recovery_moving_s=float(dt[recovery&moving].sum()),visual_recovery_holding_s=float(dt[recovery&~moving].sum()),
            route_counts=dict(routes),action_counts=dict(Counter(r['action'] for r in ss)),selected=len(ss),selected_holds=len(hh),aligned_recovery_holds=len(aligned),
            dispatch_counts=dict(Counter(r['reason'] for r,u in zip(requests,dt) if u>1e-9)),
            first_memory=scope[0] if scope else None,last_memory=scope[-1] if scope else None,
            native_path_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),net_displacement_m=float(np.linalg.norm(pose(hi)[:2]-pose(lo)[:2])),
            remaining_start=geos[leg].distance_and_heading(pose(lo)[:2]),remaining_end=geos[leg].distance_and_heading(pose(hi)[:2]))
    legs={'OUTBOUND':window(0,beacon if beacon is not None else end,'OUTBOUND')}
    if beacon is not None:legs['RETURN']=window(beacon,end,'RETURN')
    windows=[]
    for leg,row in legs.items():
        for lo in np.arange(row['start_s'],row['end_s']-1e-8,30): windows.append(dict(leg=leg,**window(float(lo),min(float(lo+30),row['end_s']),leg)))
    outbound=cells(native[(t>=0)&(t<=(beacon if beacon is not None else end)),:2]); returning=[] if beacon is None else cells(native[t>=beacon,:2])
    reverse=list(reversed(erase(outbound))); returned=erase(returning)
    common=0
    for a,b in zip(reverse,returned):
        if a!=b:break
        common+=1
    turn=None
    if beacon is not None:
        selected=[r for r in plans if r['frame']*.1>=beacon and 'selection' in r]
        first_recovery=next((r for r in selected if r.get('route_status')=='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW'),None)
        first_hold=next((r for r in selected if r['action']=='hold'),None)
        first_translation=next((float(s) for s,c in zip(qt,q) if s>=beacon and np.any(c[:2]!=0)),None)
        p=pose(beacon); wall_clearance=min(np.linalg.norm(np.maximum(np.abs(p[:2]-np.array(w['center']))-np.array(w['size'])/2,0)) for w in walls)
        turn=dict(beacon_wall_distance_base_centre_m=wall_clearance,beacon_optical_wall_ray=wall_ray(p,spec['geometry']['wall_boxes']),
            first_return_actions=[dict(time_s=r['frame']*.1,action=r['action'],route=r['route_status']) for r in selected[:20]],
            first_recovery_s=None if first_recovery is None else first_recovery['frame']*.1,
            first_hold_s=None if first_hold is None else first_hold['frame']*.1,first_translation_s=first_translation,
            initial_30s=window(beacon,min(beacon+30,end),'RETURN'))
    return dict(episode_id=ep['episode_id'],source=str(root),input_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
        beacon_s=beacon,terminal_s=end,outcome='pose_loss' if ev['source_error'] else 'timeout',shortest_round_trip_m=ep['shortest_outbound_m']+ep['shortest_return_m'],
        legs=legs,windows=windows,turnaround=turn,
        retracing=dict(outbound_cells=outbound,loop_erased_outbound=erase(outbound),return_cells=returning,loop_erased_return=returned,
            reverse_outbound_prefix_cells_matched=common,return_loop_erased_is_reverse_prefix=(returned==reverse[:len(returned)]),
            return_new_cells=[c for c in returning if c not in outbound],method='Native XY nearest 1.3-m cell centres, consecutive deduplication; loop erasure descriptive only, no route supplied to controller.'),
        inherited_hold_categories=ev['hold_categories'])

def main():
    ROOT.mkdir(exist_ok=False);output.install(ROOT)
    rows=[]
    for i in [1,5,7,8,9]:
        r=analyse(i);rows.append(r)
        with (ROOT/f'dev{i:02d}.json').open('x') as f:json.dump(r,f,indent=2)
        print(i,r['beacon_s'],r['shortest_round_trip_m'],{k:{a:round(v[a],2) for a in ['moving_s','holding_s','visual_recovery_s']} for k,v in r['legs'].items()},r['retracing'],flush=True)
    with (ROOT/'result.json').open('x') as f:json.dump(dict(rows=rows,no_physics=True,no_models=True,exploratory=True,
        time_definition='Applied nonzero command = moving (includes in-place turns); zero command = holding, including settling and vetoes. Visual recovery overlaps these two and is the latest logged planning-route state, not an exact camera-cadence latch duration.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
if __name__=='__main__':main()
