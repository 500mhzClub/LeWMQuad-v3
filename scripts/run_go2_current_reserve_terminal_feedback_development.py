"""Fixed exposed-maze test of stronger non-predictive current-state feedback."""
import hashlib
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.current_reserve_terminal_feedback_development import ReservedTerminalFeedbackMixin
from scripts import run_go2_rollout_selection_off_development as previous

runner=previous.runner
reference=previous.reference
ARMS=previous.ARMS
ROOT='go2_current_reserve_terminal_feedback_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


class ReservedTerminalRuntime(ReservedTerminalFeedbackMixin,reference.SignedLearnedRuntime):
    pass


def mark(name,value):
    if name=='launch.json':
        value=value|dict(experiment='current_reserve_terminal_feedback_v1',
            comparison='current_state_reserve_and_terminal_position_priority_without_learned_rollouts',
            actual_runtime_class='ReservedTerminalRuntime',
            rollout_off_reference_root_name=previous.ROOT.format(index=INDEX,arm=value['study_arm']),
            current_action_clearance_reserve_m=.03,
            current_clearance_required_for_translation_and_turns_m=.48,
            current_terminal_position_priority=True,
            terminal_priority_uses_future_pose=False,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'lewm/current_reserve_terminal_feedback_development.py')})
    RAW_WRITE(name,value)


def annotate(name,value):
    bind(previous.annotate,ARM=ARM,INDEX=INDEX,
        RAW_WRITE=bind(mark,INDEX=INDEX,RAW_WRITE=RAW_WRITE))(name,value)


def main():
    bind(runner.main,ROOT=ROOT,annotate=annotate,ShadowStoppingRuntime=ReservedTerminalRuntime)()


if __name__=='__main__':main()
