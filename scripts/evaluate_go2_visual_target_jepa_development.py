"""Fixed branch assay and matched training-only readouts for visual-target JEPA."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from lewm.eligible_floor_registration_development import bind
from scripts import probe_go2_jepa_latent_branch_science_development as probe
from scripts import fit_go2_frozen_motion_readout_development as original
from scripts import train_go2_visual_target_jepa_development as training

OUTPUT = original.BASE/'go2_visual_target_motion_readout_v1_attempt_001'
PLAN = Path('docs/go2_visual_target_motion_readout_plan_2026-09-17.json')


def load_parent(name, plan):
    if name=='jepa': return training.load()
    return original.load_parent(name,json.loads(original.PLAN.read_text()))


def load_readout(name):
    return bind(original.load_readout,OUTPUT=OUTPUT,PLAN=PLAN,load_parent=load_parent)(name)


def prepare_readout():
    assert not OUTPUT.exists()
    assert json.loads((training.OUTPUT/'result.json').read_text())['status']=='complete'
    plan = json.loads(original.PLAN.read_text())
    plan.update(schema='visual_target_motion_readout.v1',
        change='jepa label now visual-only target JEPA; supervised/untrained controls reproduced unchanged',
        visual_target_fit_sha256=probe.digest(training.OUTPUT/'fit.json'),
        common_training_rows_weights_ridge_penalty_and_hidden_feature_protocol=True,
        source_sha256=plan['source_sha256'] | {__file__:probe.digest(__file__)})
    probe.save(PLAN,plan); OUTPUT.mkdir()
    print('PREPARED same readout protocol for visual JEPA and unchanged controls',flush=True)


def fit_readout():
    with torch.inference_mode():
        bind(original.fit.__wrapped__,OUTPUT=OUTPUT,PLAN=PLAN,
             load_parent=load_parent,load_readout=load_readout)()


def branches(adapted=False):
    root = OUTPUT if adapted else training.OUTPUT
    fit_plan = PLAN if adapted else training.PLAN
    tag = 'visual_target_readout' if adapted else 'visual_target_raw'
    plan_path = Path(f'docs/go2_{tag}_branch_plan_2026-09-17.json')
    result_path = Path(f'docs/go2_{tag}_branch_result_2026-09-17.json')
    plan = json.loads(probe.PLAN.read_text())
    arms = ('visual_jepa','jepa','supervised_rollout','untrained')
    plan.update(arms=arms,fit_plan_sha256=probe.digest(fit_plan),
        fit_result_sha256=probe.digest(root/'result.json'),
        same_target_space_across_arms=False,
        visual_jepa_only_has_visual_target=True,
        motion_readouts_adapted=adapted,
        matched_budget_and_initialization=True)
    probe.save(plan_path,plan)

    def loader(arm):
        if arm=='visual_jepa':return load_readout('jepa') if adapted else training.load()
        return original.load_readout(arm) if adapted else original.load_parent(arm,json.loads(original.PLAN.read_text()))

    context = SimpleNamespace(PLAN=fit_plan,OUTPUT=root,load_readout=loader)
    with torch.inference_mode():
        bind(probe.run.__wrapped__,OUTPUT=root/'branch_evaluation',PLAN=plan_path,
             RESULT=result_path,ARMS=arms,fits=context)()
    from scripts import read_go2_jepa_branch_decomposition_development as decomposition
    bind(decomposition.main,OUTPUT=root/'branch_evaluation',ARMS=arms)()
    print('VISUAL_TARGET_BRANCH_EVALUATION_COMPLETE',tag,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--branches',action='store_true')
    group.add_argument('--prepare-readout',action='store_true')
    group.add_argument('--fit-readout',action='store_true')
    group.add_argument('--readout-branches',action='store_true')
    args=parser.parse_args()
    if args.branches:branches()
    elif args.prepare_readout:prepare_readout()
    elif args.fit_readout:fit_readout()
    else:branches(adapted=True)
