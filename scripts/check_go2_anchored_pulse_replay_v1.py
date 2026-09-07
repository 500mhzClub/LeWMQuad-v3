"""Read-only compatibility check on frozen physical decisions, not a new trial."""
import json
from lewm.anchored_pulse_servo_development import AnchoredPulseServo
from lewm.sensor_anchored_goal_development import AnchoredGoal
from scripts.run_go2_goal_region_pulse_servo_v1 import OUTPUT,TRIALS
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest

IDENTITIES={'launch.json':'77a038b8515b15a412c05722342168b6030549c47b77e09b012bc31d0856cf3b',
            'result.json':'05f8ba6785e23151fb69e988c71f11ff859b0e4f055b8f213cba02e31443dcdf',
            'raw_servo_audit.json':'27e43fae55c2c71a5398adbbff5d6061f4803ed6f6bde93617a2486803ddfb6b'}


def main():
    for name,h in IDENTITIES.items():assert digest(OUTPUT/name)==h
    launch=read_json(OUTPUT,'launch.json');verify(launch)
    result=read_json(OUTPUT,'result.json');audit=read_json(OUTPUT,'raw_servo_audit.json')
    assert audit['status']=='RAW_SERVO_AUDIT_PASS' and not result['absent_expected_artifacts']
    verify_bindings({str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()})
    count=0
    for c in TRIALS:
        rows=read_json(OUTPUT/c,'servo_decisions.json');first=dict(rows[0]['evidence']);now=first['decision_ns']
        # Restore only the serialized tuple at the JSON boundary. The live
        # sensor contract remains strict; no invented sensor or pose values.
        first['identity']=tuple(first['identity'])
        goal=AnchoredGoal.from_observation(first,[.4,0.],.3,identity=(0,0,0),now_ns=now)
        model=AnchoredPulseServo(goal)
        for row in rows:
            actual=model.step(row['evidence'],now_ns=row['evidence']['decision_ns'])
            actual.pop('goal')
            assert json.loads(json.dumps(actual))==row['decision'],(c,row['tick'])
        count+=len(rows)
        print('EXACT_ANCHORED_REPLAY_PASS',c,len(rows),rows[-1]['decision']['terminal'],flush=True)
    print('ALL_REPLAY_PASS',count,'same recorded poses/actions, not new physical execution or memory evidence',flush=True)


if __name__=='__main__':main()
