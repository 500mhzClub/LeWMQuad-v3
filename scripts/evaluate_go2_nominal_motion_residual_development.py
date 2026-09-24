"""Same transfer population, residual versus absolute heads on the same data."""
import torch
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_pre_switch_successor_development as previous
from scripts import nominal_motion_residual_snapshot_development as snapshot
from scripts.train_go2_nominal_motion_residual_development import OUTPUT as FITS

OUTPUT=previous.BASE/'go2_nominal_motion_residual_transfer_comparison_v1_attempt_001'


def roster():
    return [(prefix+'_'+condition,root,previous.read(root/(condition+'_fit.json')))
            for condition in previous.CONDITIONS
            for prefix,root in (('residual',FITS),('absolute',previous.FITS))]


def load(directory,record):
    if directory!=FITS:
        return previous.load(directory,record)
    return snapshot.load_snapshot(directory,record['filename'],sha256=record['sha256'],
        expected_binding=record['binding'],expected_config=record['configuration'])


def main():
    with torch.inference_mode():
        bind(previous.main.__wrapped__,OUTPUT=OUTPUT,FITS=FITS,comparison_roster=roster,load=load,
            PREDICTION_INTERPRETATION='nominal-command-composed neural residual versus absolute heads; same augmented training data')()


if __name__=='__main__':main()
