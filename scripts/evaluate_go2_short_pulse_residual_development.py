"""Fixed old and pulse transfer populations; pulse-data versus original fits."""
import json
import torch
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_pre_switch_successor_development as evaluator
from scripts import nominal_motion_residual_snapshot_development as snapshot
from scripts import prepare_go2_short_pulse_training_development as source
from scripts.train_go2_short_pulse_residual_development import OUTPUT as FITS
from scripts.train_go2_nominal_motion_residual_development import OUTPUT as OLD_FITS

OUTPUT=source.BASE/'go2_short_pulse_residual_transfer_comparison_v1_attempt_001'
TARGETS=source.BASE/'go2_short_pulse_combined_transfer_targets_v1_attempt_001'


def roster():
    return [(prefix+'_'+condition,root,evaluator.read(root/(condition+'_fit.json')))
        for condition in evaluator.CONDITIONS for prefix,root in (('pulse_augmented',FITS),('original',OLD_FITS))]


def load(directory,record):
    return snapshot.load_snapshot(directory,record['filename'],sha256=record['sha256'],
        expected_binding=record['binding'],expected_config=record['configuration'])


def main():
    terminal=evaluator.read(FITS/'result.json')
    if terminal['status']!='COMPLETE':raise ValueError('three frozen fits required')
    if TARGETS.exists() or OUTPUT.exists():raise ValueError('preserve pulse neural comparison')
    rows=evaluator.read(evaluator.TARGETS/'windows.json')
    rows += [r|dict(evaluation_population='short_pulse') for r in evaluator.read(source.PULSE/'windows.json')
        if r['data_role']=='geometry_transfer']
    TARGETS.mkdir();(TARGETS/'windows.json').write_text(json.dumps(rows)+'\n')
    with torch.inference_mode():
        bind(evaluator.main.__wrapped__,OUTPUT=OUTPUT,FITS=FITS,TARGETS=TARGETS,ROOTS=source.ROOTS,
            comparison_roster=roster,load=load,
            PREDICTION_INTERPRETATION='same residual models/budget; 900 repeated old-context draws replaced by short-pulse training draws; all original contexts retained')()


if __name__=='__main__':main()
