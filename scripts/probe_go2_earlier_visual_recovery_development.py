"""Compare camera-time warning on the exact-replayed terminal JEPA trajectory."""
import json
from copy import deepcopy
import numpy as np
from lewm.visual_support_recovery_development import LocalSupportedView
from lewm.earlier_visual_recovery_development import EarlierSupportedView
from scripts import run_go2_view_replan_repeatability_development as run


def main():
    root=run.BASE/run.root_name(4)
    read=lambda name:json.loads((root/name).read_text())
    replay=read('tracking_failure_replay_v1/result.json')
    assert replay['all_recorded_raw_poses_matched']
    poses={r['frame']:r['registered_pose'] for r in read('poses.json')}
    recorded={r['frame']:r for r in read('visual_support_recovery.json')}
    states={'original_48':LocalSupportedView(maximum_view_age_ns=None),
        'earlier_72':EarlierSupportedView(maximum_view_age_ns=None)}
    rows=[];matched=0;local_earlier=None
    terminal_reference_ns=recorded[max(recorded)]['recovery_state_at_observation']['measured_ns']
    for source in replay['rows']:
        frame=source['frame']
        if frame not in poses:continue
        pose=poses[frame];now=pose['measured_ns']
        counts=[source['feature_support'][k]['selected_features'] for k in ('primary','auxiliary')]
        row=dict(frame=frame,measured_ns=now,selected_features=counts)
        for name,state in states.items():
            active=state.advance(counts,np.asarray(pose['position_initial_body_m']),
                np.asarray(pose['rotation_initial_body_from_current_body']),now,0)
            row[name]=None if active is None else dict(trigger_ns=active['trigger_ns'],
                reference_ns=active['measured_ns'])
        if now==terminal_reference_ns:
            local_earlier=EarlierSupportedView(maximum_view_age_ns=None)
            local_earlier.__dict__=deepcopy(states['original_48'].__dict__)
        elif local_earlier is not None:
            local_earlier.advance(counts,np.asarray(pose['position_initial_body_m']),
                np.asarray(pose['rotation_initial_body_from_current_body']),now,0)
        active=None if local_earlier is None else local_earlier.active
        row['same_terminal_reference_72']=None if active is None else dict(
            trigger_ns=active['trigger_ns'],reference_ns=active['measured_ns'])
        if frame in recorded:
            original=recorded[frame]['recovery_state_at_observation'];actual=row['original_48']
            assert (original is None)==(actual is None)
            if original is not None:
                assert original['trigger_ns']==actual['trigger_ns']
                assert original['measured_ns']==actual['reference_ns']
            matched+=1
        rows.append(row)
    final=rows[-1]
    assert final['original_48']['reference_ns']==final['same_terminal_reference_72']['reference_ns']
    result=dict(schema='earlier_visual_recovery_saved_probe.v2',source_root=str(root),
        matched_original_planning_recovery_states=matched,rows=rows,
        final_original=final['original_48'],final_earlier=final['earlier_72'],
        final_same_reference_earlier=final['same_terminal_reference_72'],
        same_reference_warning_gain_ms=(final['original_48']['trigger_ns']-final['same_terminal_reference_72']['trigger_ns'])/1e6,
        full_trajectory_earlier_state_uses_different_reference=True,
        supersedes_v1_warning_gain_interpretation='v1 compared different episodes and references; its 27700 ms difference is not a same-episode warning gain',
        same_recorded_sensor_trajectory=True,native_state_used=False,
        alternative_actions_executed=False,tracking_failure_prevented_proven=False)
    run.save(root/'earlier_visual_recovery_saved_probe_v2.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
