"""Reuse the fixed 128-row maze diagnostic for the training-view intervention."""
import argparse
import torch
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_all_motion_horizon_readout_development as previous
from scripts import train_go2_maze_view_readout_development as fit

OUTPUT = fit.OUTPUT/'maze00_evaluation'
PLAN = fit.OUTPUT/'transfer_plan.json'
prepare = bind(previous.prepare,fit=fit,OUTPUT=OUTPUT,PLAN=PLAN,__file__=__file__)
main = torch.inference_mode()(bind(previous.main.__wrapped__,fit=fit,
    OUTPUT=OUTPUT,PLAN=PLAN,__file__=__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else main()
