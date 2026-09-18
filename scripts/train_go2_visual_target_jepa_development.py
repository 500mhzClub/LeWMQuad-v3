"""Matched fresh visual-target JEPA fit; retain the existing mixed-target controls."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

import cv2
import torch

from lewm.visual_target_jepa_development import VisualTargetTrainer, VisualTargetModel
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import run_go2_command_residual_lower_rate_development as baseline
from scripts import train_go2_command_history_residual_development as training
from scripts import probe_go2_jepa_latent_branch_science_development as probe

OUTPUT = training.BASE/'go2_visual_target_jepa_v1_attempt_001'
PLAN = Path('docs/go2_visual_target_jepa_plan_2026-09-17.json')


def prepare():
    assert not OUTPUT.exists()
    parent = json.loads(baseline.PLAN.read_text())
    controls = {arm:json.loads((baseline.OUTPUT/arm/'fit.json').read_text())
                for arm in ('jepa','supervised_rollout')}
    plan = dict(schema='visual_target_jepa_plan.v1', seed=parent['seed'],
        schedule_sha256=parent['schedule_sha256'], reference_sha256=parent['reference_sha256'],
        contexts=4694, draws=7200, updates=1200, batch_size=6, learning_rate=1e-4,
        latent_dim=32, ema_momentum=.99, controls=controls,
        change='only latent target: EMA RGB features plus fixed zero body/control embeddings into existing fusion',
        causal_inputs_unchanged=True, parameters_and_initialization_unchanged=True,
        future_body_control_still_available_to_common_online_variance_regularizer=True,
        common_motion_contact_losses_and_weights_unchanged=True,
        variance_penalty_not_specific_to_visual_target=True,
        final_checkpoint_only=True, independent_replicates=False,
        primary='fixed branch assay: persistence/action/scene comparisons and representation variance',
        downstream='same training-only motion readout and common recorded forecasting population',
        no_navigation_promotion_from_latent_loss=True,
        baseline_plan_sha256=probe.digest(baseline.PLAN),
        source_sha256={p:probe.digest(p) for p in (__file__, 'lewm/visual_target_jepa_development.py')},
        resources=dict(cpu_core=8, threads=1, ram_available_gib=72, output_free_gib=4.4,
                       independent_fits='reuse completed identical-budget mixed JEPA and supervised controls'))
    probe.save(PLAN,plan); OUTPUT.mkdir()
    print('PREPARED visual-target-only change, fixed 1200 updates', flush=True)


def load():
    record = json.loads((OUTPUT/'fit.json').read_text())
    assert probe.digest(OUTPUT/'model.pt') == record['checkpoint_sha256']
    checkpoint = torch.load(OUTPUT/'model.pt', map_location='cpu', weights_only=True)
    assert checkpoint['schema']=='visual_target_jepa_training_development.v1'
    model = VisualTargetModel(32)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    assert state_digest(model.state_dict()) == record['model_sha256']
    return model.eval().requires_grad_(False)


def fit():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    plan = json.loads(PLAN.read_text())
    assert probe.digest(training.SCHEDULE) == plan['schedule_sha256']
    assert probe.digest(training.REFERENCE) == plan['reference_sha256']
    for p,sha in plan['source_sha256'].items(): assert probe.digest(p)==sha
    schedule = json.loads(training.SCHEDULE.read_text()); rows = training.data.load_training_rows()
    assert len(rows)==4694 and all(r['data_role']=='train' for r in rows)
    probe.save(OUTPUT/'launch.json',dict(pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0))))
    started = time.monotonic(); trainer = None
    try:
        cache,identities = training.data.prepare(rows)
        expected = json.loads((baseline.OUTPUT/'jepa/consumed_policy_sha256.json').read_text())
        assert identities==expected
        probe.save(OUTPUT/'consumed_policy_sha256.json',identities)
        trainer = VisualTargetTrainer(seed=plan['seed'])
        assert trainer.initial_sha256==plan['controls']['jepa']['initial_sha256']
        example = training.previous.batch(cache, schedule['batches'][0])
        observed = example['targets']['future_observations']
        observed = {k:v[example['targets']['future_valid']] for k,v in observed.items()}
        # The target API must work with RGB alone, proving it cannot consume
        # future body/control. Also check replacing those tensors with NaNs.
        expected_target = trainer.model.target({'rgb':observed['rgb']})
        poisoned = observed | {k:torch.full_like(observed[k],float('nan')) for k in ('body','control')}
        torch.testing.assert_close(trainer.model.target(poisoned),expected_target,rtol=0,atol=0)
        with (OUTPUT/'updates.jsonl').open('x') as ledger:
            for step,ids in enumerate(schedule['batches'],1):
                record=trainer.step(training.previous.batch(cache,ids))
                ledger.write(json.dumps(record)+'\n')
                if step==1 or step%300==0:
                    ledger.flush();print('VISUAL_TARGET_FIT',step,'loss',round(record['loss'],5),
                                         'elapsed_s',round(time.monotonic()-started,1),flush=True)
        assert trainer.updates==1200
        checkpoint = trainer.checkpoint()
        with (OUTPUT/'model.pt').open('xb') as stream: torch.save(checkpoint,stream)
        record = dict(status='complete', updates=trainer.updates, initial_sha256=trainer.initial_sha256,
            model_sha256=checkpoint['model_sha256'], checkpoint_sha256=probe.digest(OUTPUT/'model.pt'),
            wall_s=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            latent_target_accepts_RGB_only=True, future_body_control_poisoning_has_no_target_effect=True,
            training_inputs_exactly_match_mixed_JEPA=True, final_update=record)
        probe.save(OUTPUT/'fit.json',record)
        clone = load()
        with torch.inference_mode():
            a=trainer.model.eval()(**example['inputs']); b=clone(**example['inputs'])
            for key in ('future_latents','rollout_outcomes','direct_outcomes'):
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        probe.save(OUTPUT/'result.json',dict(status='complete',records={'visual_jepa':record},
                                          reloaded_predictions_exact=True))
        print('VISUAL_TARGET_TRAINING_COMPLETE', json.dumps(record), flush=True)
    except Exception as error:
        probe.save(OUTPUT/'failure.json',dict(reason=repr(error),updates=None if trainer is None else trainer.updates))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare() if args.prepare else fit()
