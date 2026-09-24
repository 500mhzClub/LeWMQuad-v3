"""One-factor lower learning rate for the fixed command-residual experiment."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from lewm.command_history_residual_learning_development import CommandHistoryResidualTrainer
from scripts import train_go2_command_history_residual_development as previous

OUTPUT=previous.BASE/'go2_command_residual_lower_rate_matched_fits_v1_attempt_001'
PLAN=Path('docs/go2_command_residual_lower_rate_plan_2026-09-17.json')
RATE=1e-4


def prepare():
    if OUTPUT.exists():raise ValueError('preserve every attempt')
    original=json.loads(previous.PLAN.read_text())
    diagnosis=previous.OUTPUT/'training_prediction_diagnosis/result.json'
    assert json.loads(diagnosis.read_text())['contexts']==4694
    plan=original | dict(schema='command_residual_lower_rate_plan.v1',learning_rate=RATE,
        predecessor_plan=str(previous.PLAN),predecessor_learning_rate=1e-3,
        change='learning rate only: 1e-3 to 1e-4, both conditions freshly initialized',
        motivation='learned corrections degrade reference on original training inputs as well as navigation inputs',
        diagnosis_sha256=hashlib.sha256(diagnosis.read_bytes()).hexdigest(),
        architecture_reference_data_schedule_seed_losses_updates_unchanged=True,
        posthoc_residual_damping=False,checkpoint_selection=False,
        source_sha256=original['source_sha256'] | {
            __file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    previous.write(PLAN,plan);OUTPUT.mkdir();print('PREPARED fixed 1e-4 learning-rate comparison',flush=True)


def run(condition):
    bind(previous.fit,OUTPUT=OUTPUT,PLAN=PLAN,
         CommandHistoryResidualTrainer=partial(CommandHistoryResidualTrainer,learning_rate=RATE))(condition)


def context():
    return SimpleNamespace(OUTPUT=OUTPUT,PLAN=PLAN,BASE=previous.BASE,data=previous.data,
        CONDITIONS=previous.CONDITIONS,SCHEDULE=previous.SCHEDULE,write=previous.write)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--condition',choices=previous.CONDITIONS)
    group.add_argument('--evaluate',action='store_true');group.add_argument('--diagnose',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.condition:run(args.condition)
    elif args.evaluate:
        import torch
        from scripts import evaluate_go2_command_history_residual_development as evaluator
        # inference_mode is a wrapper; bind the underlying function's globals.
        with torch.inference_mode():bind(evaluator.main.__wrapped__,train=context(),OUTPUT=OUTPUT/'prediction_evaluation')()
    else:
        import torch
        from scripts import diagnose_go2_command_residual_training_error_development as diagnostic
        with torch.inference_mode():bind(diagnostic.main.__wrapped__,train=context(),OUTPUT=OUTPUT/'training_prediction_diagnosis')()
