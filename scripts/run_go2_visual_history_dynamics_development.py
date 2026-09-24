"""Change only frozen history features in the anchored visual predictor study."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from lewm.eligible_floor_registration_development import bind
from scripts import train_go2_anchored_visual_dynamics_development as previous
from scripts import evaluate_go2_anchored_visual_dynamics_development as branch
from scripts import evaluate_go2_anchored_visual_navigation_development as navigation

OUTPUT=previous.probe.fits.BASE/'go2_visual_history_dynamics_v1_attempt_001'
PLAN=Path('docs/go2_visual_history_dynamics_plan_2026-09-17.json')
NAV_PLAN=Path('docs/go2_visual_history_navigation_plan_2026-09-17.json')
NAV_RESULT=Path('docs/go2_visual_history_navigation_result_2026-09-17.json')
BRANCH_RESULT=Path('docs/go2_visual_history_dynamics_result_2026-09-17.json')


class VisualHistoryView:
    """Expose frozen EMA visual history without changing target or baseline model."""
    def __init__(self,model):self.model=model
    def encoder(self,values):return self.model.target({'rgb':values['rgb']})
    def target(self,values):return self.model.target(values)
    def state_dict(self):return self.model.state_dict()
    def __call__(self,**inputs):return self.model(**inputs)


def visual_history():return VisualHistoryView(previous.representation.load())


class ConstantVisualVelocity:
    def __call__(self,past,anchor,blocks,valid):
        velocity=past[:,-1]-past[:,-2]
        steps=torch.arange(1,9,dtype=anchor.dtype,device=anchor.device)[None,:,None]
        predicted=anchor[:,None]+steps*velocity[:,None]
        return torch.where(valid,predicted,torch.zeros_like(predicted))


def load(arm):
    if arm=='constant_visual_velocity':return ConstantVisualVelocity()
    return bind(previous.load,OUTPUT=OUTPUT)(arm)


def context():
    representation=SimpleNamespace(OUTPUT=previous.representation.OUTPUT,load=visual_history)
    return SimpleNamespace(**(vars(previous) | dict(OUTPUT=OUTPUT,PLAN=PLAN,
        representation=representation,load=load,ARMS=previous.ARMS+('constant_visual_velocity',))))


def prepare():
    assert not OUTPUT.exists()
    plan=json.loads(previous.PLAN.read_text())
    plan.update(schema='visual_history_dynamics.v1',
        context='four frozen EMA visual targets instead of mixed online observation embeddings',
        only_context_feature_source_changed=True,past_body_control_not_consumed_by_new_predictors=True,
        same_anchor_target_architecture_initialization_schedule_optimizer_and_loss=True,
        previous_fit_sha256=previous.probe.digest(previous.OUTPUT/'result.json'),
        extra_evaluation_baseline='unfitted linear extrapolation of last two visual states, no clipping',
        source_sha256=plan['source_sha256'] | {__file__:previous.probe.digest(__file__)})
    previous.probe.save(PLAN,plan);OUTPUT.mkdir()
    print('PREPARED visual-history-only factor, two matched fits',flush=True)


def fit():
    bind(previous.fit,OUTPUT=OUTPUT,PLAN=PLAN,representation=context().representation)()
    current=json.loads((OUTPUT/'result.json').read_text())
    preceding=json.loads((previous.OUTPUT/'result.json').read_text())
    assert current['initial_state_sha256']==preceding['initial_state_sha256']
    for arm in previous.ARMS:
        assert current['records'][arm]['before']==preceding['records'][arm]['before']
    previous.probe.save(OUTPUT/'comparison.json',dict(
        common_initial_parameters_and_training_normalization_exact=True,
        initial_persistence_training_errors_exact=True,
        change='frozen EMA visual history replaces frozen mixed online history'))


def branches():
    with torch.inference_mode():
        bind(branch.main.__wrapped__,training=context(),OUTPUT=OUTPUT/'branch_evaluation',RESULT=BRANCH_RESULT)()


def prepare_navigation():
    ctx=context();arms=ctx.ARMS+('persistence','original_visual_jepa')
    bind(navigation.prepare,training=ctx,OUTPUT=OUTPUT/'navigation_visual_forecasts',
         PLAN=NAV_PLAN,ARMS=arms,__file__=__file__)()


def run_navigation():
    ctx=context();arms=ctx.ARMS+('persistence','original_visual_jepa')
    with torch.inference_mode():
        bind(navigation.run.__wrapped__,training=ctx,OUTPUT=OUTPUT/'navigation_visual_forecasts',
             PLAN=NAV_PLAN,RESULT=NAV_RESULT,ARMS=arms,
             metrics=bind(navigation.metrics,ARMS=arms),__file__=__file__)()


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare','fit','branches','prepare-navigation','navigation'):
        group.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.fit:fit()
    elif args.branches:branches()
    elif args.prepare_navigation:prepare_navigation()
    else:run_navigation()
