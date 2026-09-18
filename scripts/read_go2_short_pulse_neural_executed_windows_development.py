"""Matched new/old neural predictions on fixed exposed execution windows."""
import argparse
import torch
from lewm.eligible_floor_registration_development import bind
from scripts import read_go2_nominal_residual_executed_windows_development as reader
from scripts.evaluate_go2_short_pulse_residual_development import roster,load


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('first_pair','all_pulses'),required=True)
    args=parser.parse_args()
    roots=reader.ROOT_NAMES
    if args.scope=='all_pulses':
        cohort=reader.read(reader.BASE/'go2_neural_rgb_transfer_complete_comparison_v1_attempt_001/result.json')
        roots=tuple(r['root_name'] for r in cohort['rows'])
    output=reader.BASE/f'go2_short_pulse_neural_executed_{args.scope}_v1_attempt_001'

    def selected_read(path):
        value=reader.read(path)
        if args.scope=='all_pulses' and path.name=='saved_executed_motion_forecast_evaluation_v1.json':
            value=value|dict(rows=[r for r in value['rows'] if r['group']=='translation_pulse'])
        return value

    with torch.inference_mode():
        bind(reader.main.__wrapped__,OUTPUT=output,ROOT_NAMES=roots,roster=roster,load=load,read=selected_read)()


if __name__=='__main__':main()
