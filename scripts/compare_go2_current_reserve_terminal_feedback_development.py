"""Add the stronger non-predictive baseline to the retained comparison."""
import argparse
from scripts.compare_go2_rollout_selection_off_development import compare,path
from scripts import run_go2_current_reserve_terminal_feedback_development as experiment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layout-index',type=int,choices=(0,1),required=True)
    index=parser.parse_args().layout_index;arm=experiment.ARMS[0]
    roots=dict(forecast=path(experiment.reference.ROOT.format(index=index,arm=arm)),
        instantaneous_with_predictive_checks=path(f'go2_instantaneous_waypoint_score_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'),
        nominal_current_feedback=path(experiment.previous.ROOT.format(index=index,arm=arm)),
        reserved_terminal_feedback=path(experiment.ROOT.format(index=index,arm=arm)))
    compare(index,arm,roots,path(f'go2_current_reserve_terminal_feedback_comparison_layout{index:02d}_v1_attempt_001'),
        f'Predictive and current-state feedback: exposed maze {index}')


if __name__=='__main__':main()
