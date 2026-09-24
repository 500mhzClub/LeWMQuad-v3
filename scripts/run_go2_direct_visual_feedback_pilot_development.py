"""Fixed feedback comparator, two local tasks with the same arrival readout."""
import argparse
import json
from pathlib import Path
import shutil

from lewm.direct_visual_feedback_control_development import DirectVisualFeedbackControl
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_dense_visual_arrival_pilot_development as previous

pilot = previous.pilot
NAME = 'go2_direct_visual_feedback_pilot_v1_attempt_001'
OUTPUTS = (pilot.GOAL_ROOT.parent/NAME, previous.OUTPUT.parent/NAME)
PLAN = Path('docs/go2_direct_visual_feedback_pilot_plan_2026-09-17.json')
SOURCES = previous.SOURCES+('lewm/direct_visual_feedback_control_development.py', __file__)
CASES = previous.CASES


def prepare():
    assert not PLAN.exists() and all(not p.exists() for p in OUTPUTS)
    references = [previous.OUTPUT/f'case_{i:02d}' for i in (0, 1)]
    # Full retained case 0 bounds both same-length new recordings.
    size = sum(p.stat().st_size for p in references[0].iterdir()
        if p.name != 'sealed_test.json' and not p.name.startswith('sealed_') and p.is_file())
    required = 512*1024**2+int(1.25*size)
    available = [shutil.disk_usage(p.parent).free for p in OUTPUTS]
    assert all(v > required for v in available), (available, required)
    plan = json.loads(previous.PLAN.read_text())|dict(
        source_sha256={p:pilot.base.digest(p) for p in SOURCES},
        predecessor=str(previous.OUTPUT), case_directories=[str(p/f'case_{i:02d}') for i,p in enumerate(OUTPUTS)],
        intervention='replace future-image planning with fixed direct signed visual-goal feedback; retain arrival latch',
        feedback_rule=dict(bearing_gain=1.5, final_heading_gain=-.5, turn_heading_gain=1.5,
            max_yaw_rate=.45, position_tolerance_m=.03,
            forward='nearest yaw-rate among forward/left_arc/right_arc when goal is in forward half-plane',
            turn='nearest pure turn when goal is behind or position already within tolerance',
            hold='same learned arrival latch; no other hold'),
        predictor_called=False, encoder_and_readout_unchanged=True,
        resources=dict(output_free_bytes=available, per_volume_required_bytes=required,
            available_ram_gib=73, cpu_groups=[[4,5,6,7],[8,9,10,11]], competing_experiments=0),
        output_volume='case 0 root volume, case 1 dedicated experiment volume',
        limitations=['two exposed tasks; fixed untuned feedback rule',
            'same discrete action bank may limit near-goal feedback accuracy',
            'shared initialization loads then releases unused predictor; no predictor forward call',
            'comparison isolates controller choice, not JEPA representation learning',
            'not independent full-maze, real-time or hardware evidence'])
    for p in OUTPUTS:p.mkdir()
    pilot.base.save(PLAN,plan)
    for p in OUTPUTS:pilot.base.save(p/'plan.json',plan)
    print('DIRECT_FEEDBACK_PREPARED', json.dumps(plan['case_directories']), flush=True)


def run(case):
    plan=json.loads(PLAN.read_text())
    assert pilot.base.digest(previous.fitted.RESULT)==plan['arrival_readout_fit_sha256']
    bind(pilot.run, OUTPUT=OUTPUTS[case], PLAN=PLAN, SOURCE_FILES=SOURCES,
         CASES=CASES, DenseVisualGoalControl=DirectVisualFeedbackControl)(case)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--case',type=int,choices=(0,1));a=p.parse_args()
    if a.prepare and a.case is None:prepare()
    elif not a.prepare and a.case is not None:run(a.case)
    else:p.error('prepare or execute one fresh case')
